from collections import defaultdict
from importlib.resources import files
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml

from codetiming import Timer
import typer
import xarray as xr

from mlde_utils import VariableMetadata

from mlde_data.canari_le_sprint import CanariLESprintVariableFile

from .options import DomainOption, CollectionOption
from ..moose import (
    VARIABLE_CODES,
    raw_nc_filepath,
    processed_nc_filepath,
    remove_forecast,
    remove_pressure,
)
from mlde_utils.data.coarsen import Coarsen
from mlde_utils.data.constrain import Constrain
from mlde_utils.data.diff import Diff
from mlde_utils.data.regrid import Regrid
from mlde_utils.data.remapcon import Remapcon
from mlde_utils.data.select_domain import SelectDomain
from mlde_utils.data.select_gcm_domain import SelectGCMDomain
from mlde_utils.data.shift_lon_break import ShiftLonBreak
from mlde_utils.data.sum import Sum
from mlde_utils.data.vorticity import Vorticity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


def get_variable_resolution(config, collection):
    if config["sources"]["type"] == "moose":
        if collection == CollectionOption.cpm:
            variable_resolution = "2.2km"
        elif collection == CollectionOption.gcm:
            variable_resolution = "60km"
        else:
            raise f"Unknown collection {collection}"
    elif config["sources"]["type"] == "bp":
        # assume bp sourced data is at the desired resolution already
        if collection == CollectionOption.cpm:
            variable_resolution = "2.2km-coarsened-gcm"
        elif collection == CollectionOption.gcm:
            variable_resolution = "60km"
        else:
            raise f"Unknown collection {collection}"
    elif config["sources"]["type"] == "canari-le-sprint":
        # CANARI LE Sprint data is at 60km resolution
        variable_resolution = "60km"
    else:
        raise RuntimeError(f"Unknown souce type {config['sources']['type']}")

    return variable_resolution


def get_sources(
    config,
    collection,
    year,
    data_basedir,
    domain,
    target_size,
    variable_resolution,
    target_resolution,
    ensemble_member,
):
    sources = {}

    if config["sources"]["type"] == "moose":
        if collection == CollectionOption.cpm:
            source_domain = "uk"
        elif collection == CollectionOption.gcm:
            source_domain = "global"
        else:
            raise f"Unknown collection {collection}"
        # ds = xr.open_mfdataset([raw_nc_filepath(variable=source, year=year, frequency=frequency) for source in config['sources']['moose']])
        # for source in config['sources']['moose']:
        #     if "moose_name" in VARIABLE_CODES[source]:
        #         logger.info(f"Renaming {VARIABLE_CODES[source]['moose_name']} to {source}...")
        #         ds = ds.rename({VARIABLE_CODES[source]["moose_name"]: source})

        for src_variable in config["sources"]["variables"]:
            source_nc_filepath = raw_nc_filepath(
                variable=src_variable["name"],
                year=year,
                frequency=src_variable["frequency"],
                resolution=variable_resolution,
                collection=collection.value,
                domain=source_domain,
                ensemble_member=ensemble_member,
            )
            logger.info(f"Opening {source_nc_filepath}")
            ds = xr.open_dataset(source_nc_filepath)

            if "moose_name" in VARIABLE_CODES[src_variable["name"]]:
                logger.info(
                    f"Renaming {VARIABLE_CODES[src_variable['name']]['moose_name']} to {src_variable['name']}..."
                )
                ds = ds.rename(
                    {
                        VARIABLE_CODES[src_variable["name"]][
                            "moose_name"
                        ]: src_variable["name"]
                    }
                )

            # remove forecast related coords that we don't need
            ds = remove_forecast(ds)
            # remove pressure related dims and encoding data that we don't need
            ds = remove_pressure(ds)

            sources[src_variable["name"]] = ds
    elif config["sources"]["type"] == "bp":
        for src_variable in config["sources"]["variables"]:
            source_metadata = VariableMetadata(
                data_basedir,
                frequency=src_variable["frequency"],
                domain=f"{domain.value}-{target_size}",
                resolution=f"{variable_resolution}-{target_resolution}",
                ensemble_member=ensemble_member,
                variable=src_variable["name"],
            )
            source_nc_filepath = source_metadata.filepath(year)
            logger.info(f"Opening {source_nc_filepath}")
            ds = xr.open_dataset(source_nc_filepath)

            ds = remove_pressure(ds)

            sources[src_variable["name"]] = ds
    elif config["sources"]["type"] == "canari-le-sprint":
        for src_variable in config["sources"]["variables"]:
            source_metadata = CanariLESprintVariableFile(
                frequency=src_variable["frequency"],
                ensemble_member=ensemble_member,
                variable=src_variable["name"],
                year=year,
            )
            source_nc_filepath = source_metadata.filepath
            logger.info(f"Opening {source_nc_filepath}")
            ds = xr.open_dataset(source_nc_filepath)

            ds = ds.rename(
                {
                    source_metadata.varcode: src_variable["name"],
                    "time_counter": "time",
                    "axis_nbounds": "bnds",
                }
            )
            if "lat_um_atmos_grid_t" in ds.dims:
                ds = ds.rename(
                    {
                        "lat_um_atmos_grid_t": "latitude",
                        "lon_um_atmos_grid_t": "longitude",
                        "bounds_lat_um_atmos_grid_t": "latitude_bnds",
                        "bounds_lon_um_atmos_grid_t": "longitude_bnds",
                    }
                )
            if "lat_um_atmos_grid_uv" in ds.dims:
                ds = ds.rename(
                    {
                        "lat_um_atmos_grid_uv": "latitude",
                        "lon_um_atmos_grid_uv": "longitude",
                        "bounds_lat_um_atmos_grid_uv": "latitude_bnds",
                        "bounds_lon_um_atmos_grid_uv": "longitude_bnds",
                    }
                )

            ds = ds.assign(
                latitude_longitude=xr.DataArray(
                    data=0, dims=[], coords=dict(), attrs=dict(earth_radius=6371229.0)
                )
            )
            ds[src_variable["name"]] = ds[src_variable["name"]].assign_attrs(
                grid_mapping="latitude_longitude"
            )

            sources[src_variable["name"]] = ds
    else:
        raise RuntimeError(f"Unknown souce type {config['sources']['type']}")

    logger.info(f"Combining {config['sources']}...")
    ds = xr.combine_by_coords(
        sources.values(),
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )

    return ds


@app.command()
@Timer(name="create-variable", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def create(
    config_path: Path = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = "day",
    domain: DomainOption = DomainOption.london,
    scenario="rcp85",
    scale_factor: str = typer.Option(...),
    target_resolution: str = "2.2km",
    target_size: int = 64,
    ensemble_member: str = typer.Option(...),
):
    """
    Create a new variable from moose data
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # add cli parameters to config
    config["parameters"] = {
        "frequency": frequency,
        "domain": domain.value,
        "scenario": scenario,
        "scale_factor": scale_factor,
        "target_resolution": target_resolution,
    }

    data_basedir = os.path.join(os.getenv("DERIVED_DATA"), "moose")

    collection = CollectionOption(config["sources"]["collection"])

    variable_resolution = get_variable_resolution(config, collection)

    ds = get_sources(
        config,
        collection,
        year,
        data_basedir,
        domain,
        target_size,
        variable_resolution,
        target_resolution,
        ensemble_member=ensemble_member,
    )

    for job_spec in config["spec"]:
        if job_spec["action"] == "sum":
            logger.info(f"Summing {job_spec['params']['variables']}")
            ds = Sum(**job_spec["params"]).run(ds)
        elif job_spec["action"] == "diff":
            logger.info(
                f"Difference between {job_spec['params']['left']} and {job_spec['params']['right']}"
            )
            ds = Diff(**job_spec["params"]).run(ds)
        elif job_spec["action"] == "query":
            logger.info(f"Selecting {job_spec['parameters']}")
            ds = ds.sel(**job_spec["parameters"])
        elif job_spec["action"] == "coarsen":
            if scale_factor == "gcm":
                typer.echo(f"Remapping conservatively to gcm grid...")
                variable_resolution = f"{variable_resolution}-coarsened-gcm"
                # pick the target grid based on the job spec
                # some variables use one grid, others a slightly offset one
                grid_type = job_spec["parameters"]["grid"]
                target_grid_filepath = files("mlde_utils.data").joinpath(
                    f"target_grids/60km/global/{grid_type}/moose_grid.nc"
                )
                ds = Remapcon(target_grid_filepath).run(ds)
            else:
                scale_factor = int(scale_factor)
                if scale_factor == 1:
                    typer.echo(
                        f"{scale_factor}x coarsening scale factor, nothing to do..."
                    )
                else:
                    typer.echo(f"Coarsening {scale_factor}x...")
                    variable_resolution = (
                        f"{variable_resolution}-coarsened-{scale_factor}x"
                    )
                    ds, orig_ds = Coarsen(scale_factor=scale_factor).run(ds)
        elif job_spec["action"] == "shift_lon_break":
            ds = ShiftLonBreak().run(ds)
        elif job_spec["action"] == "regrid_to_target":
            if target_resolution != variable_resolution:
                typer.echo(f"Regridding to target resolution...")
                target_grid_path = files("mlde_utils.data").joinpath(
                    f"target_grids/{target_resolution}/uk/moose_grid.nc"
                )
                kwargs = job_spec.get("parameters", {})
                ds = Regrid(
                    target_grid_path, variables=[config["variable"]], **kwargs
                ).run(ds)
        elif job_spec["action"] == "vorticity":
            typer.echo(f"Computing vorticity...")
            ds = Vorticity(**job_spec["parameters"]).run(ds)
        elif job_spec["action"] == "select-subdomain":
            typer.echo(f"Select {domain.value} subdomain...")
            ds = SelectDomain(subdomain=domain.value, size=target_size).run(ds)
        elif job_spec["action"] == "select-gcm-subdomain":
            typer.echo(f"Select {domain.value} GCM subdomain...")
            ds = SelectGCMDomain(subdomain=domain.value, size=target_size).run(ds)
        elif job_spec["action"] == "constrain":
            typer.echo(f"Filtering...")
            ds = Constrain(query=job_spec["query"]).run(ds)
        elif job_spec["action"] == "rename":
            typer.echo(f"Renaming...")
            ds = ds.rename(job_spec["mapping"])
        else:
            raise RuntimeError(f"Unknown action {job_spec['action']}")

    # assign any attributes from config file
    ds[config["variable"]] = ds[config["variable"]].assign_attrs(config["attrs"])

    grid_mapping = ds[config["variable"]].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        assert len(ds.grid_latitude) == target_size
        assert len(ds.grid_longitude) == target_size
    elif grid_mapping == "latitude_longitude":
        assert len(ds.latitude) == target_size
        assert len(ds.longitude) == target_size
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")

    # there should be no missing values in this dataset
    assert ds[config["variable"]].isnull().sum().values.item() == 0

    output_metadata = VariableMetadata(
        data_basedir,
        frequency=frequency,
        domain=f"{domain.value}-{target_size}",
        resolution=f"{variable_resolution}-{target_resolution}",
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=config["variable"],
    )

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)

    ds[config["variable"]].encoding.update(dict(zlib=True, complevel=5))
    ds.to_netcdf(output_metadata.filepath(year))
    with open(
        os.path.join(output_metadata.dirpath(), f"{config['variable']}-{year}.yml"), "w"
    ) as f:
        yaml.dump(config, f)


def run_cmd(cmd):
    logger.debug(f"Running {cmd}")
    output = subprocess.run(cmd, capture_output=True, check=False)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    output.check_returncode()


@app.command()
@Timer(name="xfer-variable", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def xfer(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    ensemble_member: str = typer.Option(...),
    frequency: str = "day",
    domain: DomainOption = DomainOption.london,
    collection: CollectionOption = typer.Option(...),
    resolution: str = typer.Option(...),
    target_size: int = 64,
):
    # TODO re-write xfer in Python
    jasmin_filepath = processed_nc_filepath(
        variable=variable,
        year=year,
        frequency=frequency,
        domain=f"{domain.value}-{target_size}",
        resolution=resolution,
        collection=collection.value,
        ensemble_member=ensemble_member,
    )
    bp_filepath = processed_nc_filepath(
        variable=variable,
        year=year,
        frequency=frequency,
        domain=f"{domain.value}-{target_size}",
        resolution=resolution,
        collection=collection.value,
        base_dir="/user/work/vf20964",
        ensemble_member=ensemble_member,
    )

    file_xfer_cmd = [
        # TODO: don't rely on hardcoded absolute path
        f"{os.getenv('HOME')}/code/mlde-data/bin/moose/xfer-script-direct",
        jasmin_filepath,
        bp_filepath,
    ]
    # TODO: also transfer to config used for the variable
    # config_xfer_cmd = []
    run_cmd(file_xfer_cmd)


def check_nans(ds, var):
    return ds[var].isnull().sum().values.item() == 0


def check_dims(ds, var):
    grid_mapping = ds[var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[var].dims) == [
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[var].dims) == [
            "time",
            "latitude",
            "longitude",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_forecast_encoding(ds, var):
    if "coordinates" in ds[var].encoding and (
        re.match(
            "(realization|forecast_period|forecast_reference_time) ?",
            ds[var].encoding["coordinates"],
        )
        is not None
    ):
        return False
    return True


def check_forecast_vars(ds, var):
    for v in ds.variables:
        if v in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "forecast_period_bnds",
        ]:
            return False
    return True


def check_pressure_encoding(ds, var):
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
        ):
            return False
    return True


def check_pressure_vars(ds, var):
    for v in ds.variables:
        if v in ["pressure"]:
            return False
    return True


def check_grid_vars(ds, var):
    grid_mapping = ds[var].attrs["grid_mapping"]
    meta_vars = [
        grid_mapping,
    ]
    if grid_mapping == "rotated_latitude_longitude":
        meta_vars.extend(["grid_latitude_bnds", "grid_longitude_bnds"])

    for mvar in meta_vars:
        if ("ensemble_member" in ds[mvar].dims) or ("time" in ds[mvar].dims):
            return False
    return True


def check_time_bnds(ds, var):
    return "ensemble_member" not in ds["time_bnds"].dims


@app.command()
def validate(
    variable: str = typer.Argument("all"), ensemble_member: str = typer.Argument("all")
):
    domain_res_vars = {
        "birmingham-64": {
            "2.2km-coarsened-gcm-2.2km-coarsened-4x": [
                "psl",
                # "tempgrad500250",
                # "tempgrad700500",
                # "tempgrad850700",
                # "tempgrad925850",
                "vorticity250",
                "vorticity500",
                "vorticity700",
                "vorticity850",
                "vorticity925",
                "spechum250",
                "spechum500",
                "spechum700",
                "spechum850",
                "spechum925",
                "temp250",
                "temp500",
                "temp700",
                "temp850",
                "temp925",
                # "pr",
                "linpr",
            ],
            "2.2km-coarsened-4x-2.2km-coarsened-4x": [
                "pr",
                "relhum150cm",
                "tmean150cm",
            ],
            "60km-2.2km-coarsened-4x": [
                "psl",
                # "tempgrad500250",
                # "tempgrad700500",
                # "tempgrad850700",
                # "tempgrad925850",
                "vorticity250",
                "vorticity500",
                "vorticity700",
                "vorticity850",
                "vorticity925",
                "spechum250",
                "spechum500",
                "spechum700",
                "spechum850",
                "spechum925",
                "temp250",
                "temp500",
                "temp700",
                "temp850",
                "temp925",
                "linpr",
                "pr",
            ],
        },
        "birmingham-9": {"60km-60km": ["pr"], "2.2km-coarsened-gcm-60km": ["pr"]},
    }

    years = list(range(1981, 2001)) + list(range(2021, 2041)) + list(range(2061, 2081))

    ensemble_members = [
        "01",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "15",
    ]

    for domain, res_variables in domain_res_vars.items():
        for res, variables in res_variables.items():
            for em in ensemble_members:
                if (ensemble_member != "all") and (ensemble_member != em):
                    continue
                for var in variables:
                    if (variable != "all") and (variable != var):
                        continue
                    sys.stdout.write("\033[K")
                    print(
                        f"Checking {var} of {em} over {domain} at {res}",
                        end="\r",
                    )

                    bad_years = defaultdict(set)
                    for year in years:
                        var_meta = VariableMetadata(
                            f"{os.getenv('DERIVED_DATA')}/moose",
                            variable=var,
                            frequency="day",
                            domain=domain,
                            resolution=res,
                            ensemble_member=em,
                        )

                        try:
                            ds = xr.load_dataset(var_meta.filepath(year))
                        except FileNotFoundError:
                            bad_years["no file"].add(year)
                            continue

                        # check for NaNs
                        if not check_nans(ds, var):
                            bad_years["NaNs"].add(year)

                        # check dims
                        if not check_dims(ds, var):
                            bad_years["bad dimensions"].add(year)

                        # check for forecast related metadata (should have been stripped)
                        if not check_forecast_encoding(ds, var):
                            bad_years["forecast_encoding"].add(year)
                        if not check_forecast_vars(ds, var):
                            bad_years["forecast_vars"].add(year)
                        # check for pressure related metadata (should have been stripped)
                        if not check_pressure_encoding(ds, var):
                            bad_years["pressure_encoding"].add(year)
                        if not check_pressure_vars(ds, var):
                            bad_years["pressure_vars"].add(year)
                        # check grid and time vars
                        if not check_grid_vars(ds, var):
                            bad_years["grid_meta_vars"].add(year)
                        if not check_time_bnds(ds, var):
                            bad_years["time_bnds"].add(year)

                    # report findings
                    for reason, error_years in bad_years.items():
                        if len(error_years) > 0:
                            print(
                                f"Failed '{reason}': {var} over {domain} of {em} at {res} for {len(error_years)}\n{sorted(error_years)}"
                            )
