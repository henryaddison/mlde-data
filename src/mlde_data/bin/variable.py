from codetiming import Timer
from collections import defaultdict
from importlib.resources import files
import logging
from mlde_data import DERIVED_DATA, MOOSE_DATA
from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter
from mlde_data.variable import validation, load_config
from mlde_utils import VariableMetadata
import os
import pandas as pd
from pathlib import Path
import sys
import typer
from typing import List
import xarray as xr
import yaml


from mlde_data.bin.options import CollectionOption, DomainOption
from mlde_data.moose import (
    VARIABLE_CODES,
    remove_forecast,
    remove_pressure,
)
from mlde_utils.data import get_action

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


def get_resolution(srcs_config, collection):
    if srcs_config["type"] == "moose":
        if collection == CollectionOption.cpm:
            data_resolution = "2.2km"
            grid_resolution = "2.2km"
        elif collection == CollectionOption.gcm:
            data_resolution = "60km"
            grid_resolution = "60km"
        else:
            raise f"Unknown collection {collection}"
    elif srcs_config["type"] == "local":
        # assume local sourced data is pre-processed so resolution must be specified in config
        data_resolution = srcs_config["data_resolution"]
        grid_resolution = srcs_config["grid_resolution"]
    elif srcs_config["type"] == "canari-le-sprint":
        # CANARI LE Sprint data is at 60km resolution
        data_resolution = "60km"
        grid_resolution = "60km"

    else:
        raise RuntimeError(f"Unknown souce type {srcs_config['type']}")

    return data_resolution, grid_resolution


def get_sources(
    srcs_config,
    collection,
    year,
    data_basedir,
    ensemble_member,
):
    sources = {}

    data_resolution, grid_resolution = get_resolution(srcs_config, collection)
    frequency = srcs_config["frequency"]
    scenario = "rcp85"

    if srcs_config["type"] == "moose":
        if collection == CollectionOption.cpm:
            source_domain = "uk"
        elif collection == CollectionOption.gcm:
            source_domain = "global"
        else:
            raise f"Unknown collection {collection}"

        data_resolution = data_resolution

        for src_variable in srcs_config["variables"]:
            source_nc_filepath = VariableMetadata(
                base_dir=MOOSE_DATA,
                variable=src_variable["name"],
                frequency=frequency,
                domain=source_domain,
                resolution=data_resolution,
                ensemble_member=ensemble_member,
                scenario=scenario,
                collection=collection.value,
            ).filepath(year)
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
    elif srcs_config["type"] == "local":
        source_domain = srcs_config["domain"]
        for src_variable in srcs_config["variables"]:
            source_metadata = VariableMetadata(
                data_basedir,
                frequency=frequency,
                resolution=f"{data_resolution}-{grid_resolution}",
                scenario=scenario,
                domain=source_domain,
                ensemble_member=ensemble_member,
                variable=src_variable["name"],
                collection=collection.value,
            )
            source_nc_filepath = source_metadata.filepath(year)
            logger.info(f"Opening {source_nc_filepath}")
            ds = xr.open_dataset(source_nc_filepath)

            ds = remove_pressure(ds)

            sources[src_variable["name"]] = ds
    elif srcs_config["type"] == "canari-le-sprint":
        data_resolution, grid_resolution = get_resolution(srcs_config, collection)
        source_domain = "global"
        for src_variable in srcs_config["variables"]:
            source_metadata = CanariLESprintVariableAdapter(
                frequency=frequency,
                ensemble_member=ensemble_member,
                variable=src_variable["name"],
                year=year,
            )

            sources[src_variable["name"]] = source_metadata.open().load()
    else:
        raise RuntimeError(f"Unknown souce type {srcs_config['type']}")

    logger.info(f"Combining {srcs_config}...")
    ds = xr.combine_by_coords(
        sources.values(),
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )

    return ds, data_resolution, grid_resolution, frequency, source_domain


def _process(
    ds,
    config,
    current_domain,
    current_data_resolution,
    current_grid_resolution,
    current_frequency,
):
    for job_spec in config["spec"]:
        if job_spec["action"] == "sum":
            logger.info(f"Summing {job_spec['params']['variables']}")
            ds = get_action(job_spec["action"])(**job_spec["params"])(ds)
        elif job_spec["action"] == "diff":
            logger.info(
                f"Difference between {job_spec['params']['left']} and {job_spec['params']['right']}"
            )
            ds = get_action(job_spec["action"])(**job_spec["params"])(ds)
        elif job_spec["action"] == "query":
            logger.info(f"Selecting {job_spec['parameters']}")
            ds = ds.sel(**job_spec["parameters"])
        elif job_spec["action"] == "drop-variables":
            logger.info(f"Dropping variables {job_spec['variables']}")
            ds = ds.drop_vars(job_spec["variables"])
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
            current_frequency = "day"
        elif job_spec["action"] == "coarsen":
            ds, current_data_resolution, current_grid_resolution = get_action(
                job_spec["action"]
            )(**job_spec["parameters"])(
                ds, current_data_resolution, current_grid_resolution
            )
        elif job_spec["action"] == "shift_lon_break":
            ds = get_action(job_spec["action"])()(ds)
        elif job_spec["action"] == "regrid_to_target":
            # this assumes mapping to a target grid of higher resolution than resolution of the data
            desired_grid_resolution = job_spec["parameters"]["target_grid_resolution"]
            if desired_grid_resolution != current_grid_resolution:
                typer.echo(f"Regridding to target resolution...")
                target_grid_path = files("mlde_utils.data").joinpath(
                    f"target_grids/{desired_grid_resolution}/uk/moose_grid.nc"
                )
                kwargs = job_spec.get("parameters", {})
                ds = get_action(job_spec["action"])(
                    target_grid_path, variables=[config["variable"]], **kwargs
                )(ds)
                current_grid_resolution = desired_grid_resolution
                current_domain = "uk"
        elif job_spec["action"] == "vorticity":
            typer.echo(f"Computing vorticity...")
            action = get_action(job_spec["action"])
            ds = action(**job_spec["parameters"])(ds)
        elif job_spec["action"] == "select-subdomain":
            current_domain = (
                f"{job_spec['parameters']['domain']}-{job_spec['parameters']['size']}"
            )
            typer.echo(f"Select {current_domain} subdomain...")
            ds = get_action(job_spec["action"])(**job_spec["parameters"])(ds)
        elif job_spec["action"] == "constrain":
            typer.echo(f"Filtering...")
            ds = get_action(job_spec["action"])(query=job_spec["query"])(ds)
        elif job_spec["action"] == "rename":
            typer.echo(f"Renaming...")
            ds = ds.rename(job_spec["mapping"])
        else:
            raise RuntimeError(f"Unknown action {job_spec['action']}")

    # assign any attributes from config file
    ds[config["variable"]] = ds[config["variable"]].assign_attrs(config["attrs"])

    return (
        ds,
        current_data_resolution,
        current_grid_resolution,
        current_domain,
        current_frequency,
    )


def _save(ds, config, path, year):
    logger.info(f"Saving data to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds[config["variable"]].encoding.update(dict(zlib=True, complevel=5))
    ds.to_netcdf(path)
    with open(
        os.path.join(os.path.dirname(path), f"{config['variable']}-{year}.yml"), "w"
    ) as f:
        yaml.dump(config, f)


@app.command()
@Timer(name="create-variable", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def create(
    config_path: Path = typer.Option(...),
    scenario="rcp85",
    ensemble_member: str = typer.Option(...),
    year: int = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    size: int = typer.Option(...),
):
    """
    Create a variable file in project form from source data
    """
    config = load_config(
        config_path, scale_factor=scale_factor, domain=domain.value, size=size
    )

    collection = CollectionOption(config["sources"]["collection"])
    src_type = config["sources"]["type"]

    data_basedir: Path = DERIVED_DATA / src_type

    ds, data_resolution, grid_resolution, frequency, domain = get_sources(
        config["sources"],
        collection,
        year,
        data_basedir,
        ensemble_member=ensemble_member,
    )

    ds, data_resolution, grid_resolution, domain, frequency = _process(
        ds,
        config,
        domain,
        data_resolution,
        grid_resolution,
        frequency,
    )

    if frequency == "day":
        # there should be 360 days in the dataset
        assert len(ds.time) == 360

    # there should be no missing values in this dataset
    assert ds[config["variable"]].isnull().sum().values.item() == 0

    output_metadata = VariableMetadata(
        data_basedir,
        frequency=frequency,
        domain=domain,
        resolution=f"{data_resolution}-{grid_resolution}",
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=config["variable"],
        collection=collection,
    )

    _save(ds, config, output_metadata.filepath(year), year)


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
                                f"{DERIVED_DATA/source}",
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
