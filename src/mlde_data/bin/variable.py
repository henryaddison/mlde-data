from collections import defaultdict
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

from ml_downscaling_emulator import VariableMetadata
from ml_downscaling_emulator.bin import DomainOption, CollectionOption
from ml_downscaling_emulator.data.moose import (
    VARIABLE_CODES,
    raw_nc_filepath,
    processed_nc_filepath,
    remove_forecast,
    remove_pressure,
)
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.constrain import Constrain
from ml_downscaling_emulator.preprocessing.diff import Diff
from ml_downscaling_emulator.preprocessing.regrid import Regrid
from ml_downscaling_emulator.preprocessing.remapcon import Remapcon
from ml_downscaling_emulator.preprocessing.resample import Resample
from ml_downscaling_emulator.preprocessing.select_domain import SelectDomain
from ml_downscaling_emulator.preprocessing.sum import Sum
from ml_downscaling_emulator.preprocessing.vorticity import Vorticity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


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

    sources = {}

    if config["sources"]["type"] == "moose":
        if collection == CollectionOption.cpm:
            variable_resolution = "2.2km"
            source_domain = "uk"
        elif collection == CollectionOption.gcm:
            variable_resolution = "60km"
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

            sources[src_variable["name"]] = ds
    elif config["sources"]["type"] == "bp":
        # assume bp sourced data is at the desired resolution already
        if collection == CollectionOption.cpm:
            variable_resolution = "2.2km-coarsened-gcm"
        elif collection == CollectionOption.gcm:
            variable_resolution = "60km"
        else:
            raise f"Unknown collection {collection}"
        for src_variable in config["sources"]["variables"]:
            source_metadata = VariableMetadata(
                data_basedir,
                frequency=src_variable["frequency"],
                domain=f"{domain.value}-{target_size}",
                resolution=f"{variable_resolution}-{target_resolution}",
                ensemble_member="01",
                variable=src_variable["name"],
            )
            source_nc_filepath = source_metadata.filepath(year)
            logger.info(f"Opening {source_nc_filepath}")
            ds = xr.open_dataset(source_nc_filepath)

            ds = remove_pressure(ds)

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

    for job_spec in config["spec"]:
        if job_spec["action"] == "sum":
            logger.info(f"Summing {job_spec['variables']}")
            ds = Sum(job_spec["variables"], config["variable"]).run(ds)
            ds[config["variable"]] = ds[config["variable"]].assign_attrs(
                config["attrs"]
            )
        if job_spec["action"] == "diff":
            logger.info(
                f"Difference between {job_spec['params']['left']} and {job_spec['params']['right']}"
            )
            ds = Diff(**job_spec["params"]).run(ds)
            ds[config["variable"]] = ds[config["variable"]].assign_attrs(
                config["attrs"]
            )
        elif job_spec["action"] == "coarsen":
            if scale_factor == "gcm":
                typer.echo(f"Remapping conservatively to gcm grid...")
                variable_resolution = f"{variable_resolution}-coarsened-gcm"
                # pick the target grid based on the job spec
                # some variables use one grid, others a slightly offset one
                grid_type = job_spec["parameters"]["grid"]
                target_grid_filepath = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "utils",
                    "target-grids",
                    "60km",
                    "global",
                    grid_type,
                    "moose_grid.nc",
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
        elif job_spec["action"] == "regrid_to_target":
            if target_resolution != variable_resolution:
                typer.echo(f"Regridding to target resolution...")
                target_grid_filepath = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "utils",
                    "target-grids",
                    target_resolution,
                    "uk",
                    "moose_grid.nc",
                )
                kwargs = job_spec.get("parameters", {})
                ds = Regrid(
                    target_grid_filepath, variables=[config["variable"]], **kwargs
                ).run(ds)
        elif job_spec["action"] == "vorticity":
            typer.echo(f"Computing vorticity...")
            ds = Vorticity(**job_spec["parameters"]).run(ds)
        elif job_spec["action"] == "select-subdomain":
            typer.echo(f"Select {domain.value} subdomain...")
            ds = SelectDomain(subdomain=domain.value, size=target_size).run(ds)
        elif job_spec["action"] == "constrain":
            typer.echo(f"Filtering...")
            ds = Constrain(query=job_spec["query"]).run(ds)
        elif job_spec["action"] == "rename":
            typer.echo(f"Renaming...")
            ds = ds.rename(job_spec["mapping"])
        else:
            raise f"Unknown action {job_spec['action']}"

    assert len(ds.grid_latitude) == target_size
    assert len(ds.grid_longitude) == target_size

    # there should be no missing values in this dataset
    assert ds[config["variable"]].isnull().sum().values.item() == 0

    output_metadata = VariableMetadata(
        data_basedir,
        frequency=frequency,
        domain=f"{domain.value}-{target_size}",
        resolution=f"{variable_resolution}-{target_resolution}",
        ensemble_member="01",
        variable=config["variable"],
    )

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)
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
    )
    bp_filepath = processed_nc_filepath(
        variable=variable,
        year=year,
        frequency=frequency,
        domain=f"{domain.value}-{target_size}",
        resolution=resolution,
        collection=collection.value,
        base_dir="/user/work/vf20964",
    )

    file_xfer_cmd = [
        f"{os.getenv('HOME')}/code/ml-downscaling-emulation/moose-etl/xfer-script-direct",
        jasmin_filepath,
        bp_filepath,
    ]
    config_xfer_cmd = []
    run_cmd(file_xfer_cmd)


@app.command()
def validate():
    domain_res_vars = {
        "birmingham-64": {
            "2.2km-coarsened-gcm-2.2km-coarsened-4x": [
                "psl",
                "tempgrad500250",
                "tempgrad700500",
                "tempgrad850700",
                "tempgrad925850",
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
                "pr",
            ],
            "60km-2.2km-coarsened-4x": [
                "psl",
                "tempgrad500250",
                "tempgrad700500",
                "tempgrad850700",
                "tempgrad925850",
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
                "pr",
                "lin-pr",
            ],
            "2.2km-coarsened-4x-2.2km-coarsened-4x": [
                "pr",
            ],
            "2.2km-coarsened-gcm-2.2km": ["pr", "vorticity850"],
            "60km-2.2km": ["pr", "vorticity850"],
        },
        "london-64": {
            "2.2km-2.2km": ["pr", "vorticity850"],
            "60km-2.2km": ["pr", "vorticity850"],
            "60km-2.2km-coarsened-4x": ["pr", "vorticity850"],
            "2.2km-coarsened-8x-2.2km": ["vorticity850"],
            "2.2km-coarsened-gcm-2.2km": ["vorticity850"],
        },
        "birmingham-256": {
            "2.2km-2.2km": ["pr"],
            "2.2km-coarsened-gcm-2.2km": ["vorticity850"],
            "60km-2.2km": ["pr", "vorticity850"],
        },
    }

    years = list(range(1981, 2001)) + list(range(2021, 2041)) + list(range(2061, 2081))

    for domain, res_variables in domain_res_vars.items():
        for res, variables in res_variables.items():
            for var in variables:
                sys.stdout.write("\033[K")
                print(f"Checking {var} over {domain} at {res}", end="\r")

                bad_years = defaultdict(set)
                for year in years:
                    var_meta = VariableMetadata(
                        os.getenv("MOOSE_DERIVED_DATA"),
                        variable=var,
                        frequency="day",
                        domain=domain,
                        resolution=res,
                    )

                    try:
                        ds = xr.load_dataset(var_meta.filepath(year))
                    except FileNotFoundError:
                        bad_years["no file"].add(year)
                        continue

                    nan_count = ds[var].isnull().sum().values.item()

                    if nan_count > 0:
                        bad_years["NaNs"].add(year)

                    # check for forecast related metadata (should have been stripped)
                    for v in ds.variables:
                        if "coordinates" in ds[v].encoding and (
                            re.match(
                                "(realization|forecast_period|forecast_reference_time) ?",
                                ds[v].encoding["coordinates"],
                            )
                            is not None
                        ):
                            bad_years["forecast_encoding"].add(year)
                        if v in [
                            "forecast_period",
                            "forecast_reference_time",
                            "realization",
                            "forecast_period_bnds",
                        ]:
                            bad_years["forecast_vars"].add(year)

                    # check for pressure related metadata (should have been stripped)
                    for v in ds.variables:
                        if "coordinates" in ds[v].encoding and (
                            re.match("(pressure) ?", ds[v].encoding["coordinates"])
                            is not None
                        ):
                            bad_years["pressure_encoding"].add(year)
                        if v in ["pressure"]:
                            bad_years["pressure_vars"].add(year)

                # report findings
                for reason, error_years in bad_years.items():
                    if len(error_years) > 0:
                        print(
                            f"Failed '{reason}': {var} over {domain} at {res} for {len(error_years)}\n{sorted(error_years)}"
                        )
