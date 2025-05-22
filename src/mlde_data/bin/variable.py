from collections import defaultdict
from importlib.resources import files
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import List
import pandas as pd
import yaml

from codetiming import Timer
import typer
import xarray as xr

from mlde_utils import VariableMetadata

from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter
from mlde_data.variable import validation

from .options import DomainOption, CollectionOption
from ..moose import (
    VARIABLE_CODES,
    raw_nc_filepath,
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
            source_metadata = CanariLESprintVariableAdapter(
                frequency=src_variable["frequency"],
                ensemble_member=ensemble_member,
                variable=src_variable["name"],
                year=year,
            )

            sources[src_variable["name"]] = source_metadata.open().load()
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


def _process(
    ds,
    config,
    variable_resolution,
    target_resolution,
    domain,
    scale_factor,
    target_size,
):
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
        elif job_spec["action"] == "resample":
            logger.info(f"Resampling {job_spec['parameters']}")
            new_bounds = (
                ds["time_bnds"].isel(bnds=0).resample(**job_spec["parameters"]).min()
            )
            new_bounds = xr.concat(
                [new_bounds, new_bounds + pd.Timedelta(days=1)], dim="bnds"
            )
            new_bounds.assign_attrs(ds["time_bnds"].attrs)
            new_bounds.encoding = ds["time_bnds"].encoding

            ds = ds.resample(**job_spec["parameters"]).mean()
            ds["time_bnds"] = new_bounds
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

    return ds, variable_resolution


@app.command()
@Timer(name="create-variable", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def create(
    config_path: Path = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = typer.Option(...),
    domain: DomainOption = DomainOption.london,
    scenario="rcp85",
    scale_factor: str = typer.Option(...),
    target_resolution: str = typer.Option(...),
    target_size: int = typer.Option(...),
    ensemble_member: str = typer.Option(...),
):
    """
    Create a variable file in project form from source data
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

    collection = CollectionOption(config["sources"]["collection"])
    src_type = config["sources"]["type"]

    data_basedir: Path = os.path.join(os.getenv("DERIVED_DATA"), src_type)

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

    ds, variable_resolution = _process(
        ds,
        config,
        variable_resolution,
        target_resolution,
        domain,
        scale_factor,
        target_size,
    )

    grid_mapping = ds[config["variable"]].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        assert len(ds.grid_latitude) == target_size
        assert len(ds.grid_longitude) == target_size
    elif grid_mapping == "latitude_longitude":
        assert len(ds.latitude) == target_size
        assert len(ds.longitude) == target_size
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")

    if frequency == "day":
        assert len(ds.time) == 360

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
        collection=collection,
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
def validate(
    years: List[int],
    source: str = "moose",
    collection: str = "land-cpm",
    variable: str = "all",
    ensemble_member: str = "all",
):
    frequency = "day"

    if 0 in years:
        years.remove(0)
        years.extend(validation.YEARS[source])

    for domain, res_variables in validation.DOMAIN_RES_VARS[source][collection].items():
        for res, variables in res_variables.items():
            for em in validation.ENSEMBLE_MEMBERS[source][collection]:
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

                    for scenario in validation.SCENARIOS[source]:
                        bad_years = defaultdict(set)
                        for year in years:
                            var_meta = VariableMetadata(
                                f"{os.getenv('DERIVED_DATA')}/{source}",
                                variable=var,
                                frequency=frequency,
                                domain=domain,
                                resolution=res,
                                ensemble_member=em,
                                collection=collection,
                                scenario=scenario,
                            )
                            for error in validation.validate(var_meta, year):
                                bad_years[error].add(year)

                        # report findings
                        sys.stdout.write("\033[K")
                        for reason, error_years in bad_years.items():
                            if len(error_years) > 0:
                                print(
                                    f"Failed '{reason}': {var} over {domain} of {em} in {scenario} at {res} for {len(error_years)}\n{sorted(error_years)}"
                                )

                        if not any(map(lambda s: len(s) > 0, bad_years.values())):
                            print(
                                f"Passed validation: {var} over {domain} of {em} in {scenario} at {res}"
                            )
