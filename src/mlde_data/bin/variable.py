from codetiming import Timer
from collections import defaultdict
from dataclasses import dataclass
import logging
from mlde_utils import RAW_MOOSE_VARIABLES_PATH, DERIVED_VARIABLES_PATH
from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter
from mlde_data.variable import validation, load_config
from mlde_utils import VariableMetadata
import os
from pathlib import Path
import sys
import typer
from typing import List
import xarray as xr
import yaml


from mlde_data.options import CollectionOption, DomainOption
from mlde_data.moose import (
    VARIABLE_CODES,
    open_pp_data,
    remove_forecast,
    remove_pressure,
)
from mlde_data.actions import get_action

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


@dataclass(frozen=True)
class SourceConfig:
    type: str
    collection: str
    frequency: str
    variable: str


def get_resolution(src_config: SourceConfig) -> str:
    collection = CollectionOption(src_config.collection)
    if src_config.type == "moose":
        if collection == CollectionOption.cpm:
            resolution = "2.2km"
        elif collection == CollectionOption.gcm:
            resolution = "60km"
        else:
            raise f"Unknown collection {collection}"
    elif src_config.type == "local":
        # assume local sourced data is pre-processed so resolution must be specified in config
        resolution = src_config.resolution
    elif src_config.type == "canari-le-sprint":
        # CANARI LE Sprint data is at 60km resolution
        resolution = "60km"

    else:
        raise RuntimeError(f"Unknown souce type {src_config.type}")

    return resolution


def get_source_domain(src_config: SourceConfig) -> str:
    collection = CollectionOption(src_config.collection)
    if src_config.type == "moose":
        if collection == CollectionOption.cpm:
            domain = "uk"
        elif collection == CollectionOption.gcm:
            domain = "global"
        else:
            raise RuntimeError(f"Unknown collection {collection}")
    elif src_config.type == "local":
        domain = src_config.domain
    elif src_config.type == "canari-le-sprint":
        domain = "global"
    else:
        raise RuntimeError(f"Unknown souce type {src_config.type}")

    return domain


def open_local_source_variable(
    src_variable: str,
    year: int,
    frequency: str,
    scenario: str,
    resolution: str,
    ensemble_member: str,
    domain: str,
    collection: str,
    base_dir: Path,
) -> xr.Dataset:
    source_metadata = VariableMetadata(
        base_dir=base_dir,
        frequency=frequency,
        resolution=resolution,
        scenario=scenario,
        domain=domain,
        ensemble_member=ensemble_member,
        variable=src_variable,
        collection=collection,
    )
    source_nc_filepath = source_metadata.filepath(year)
    logger.info(f"Opening {source_nc_filepath}")
    ds = xr.open_dataset(source_nc_filepath)

    ds = remove_pressure(ds)

    return ds


def open_moose_source_variable(
    src_variable: str,
    year: int,
    frequency: str,
    scenario: str,
    resolution: str,
    ensemble_member: str,
    domain: str,
    collection: str,
    base_dir: Path,
) -> xr.Dataset:
    logger.info(f"Opening {src_variable} moose extract...")
    ds = open_pp_data(
        base_dir=base_dir / "pp",
        collection=CollectionOption(collection),
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=src_variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
        year=year,
    )

    if "moose_name" in VARIABLE_CODES[src_variable]:
        logger.info(
            f"Renaming {VARIABLE_CODES[src_variable]['moose_name']} to {src_variable}..."
        )
        ds = ds.rename({VARIABLE_CODES[src_variable]["moose_name"]: src_variable})
    # remove forecast related coords that we don't need
    ds = remove_forecast(ds)

    return ds


def open_canari_le_sprint_source_variable(
    src_variable: str,
    year: int,
    frequency: str,
    scenario: str,
    resolution: str,
    ensemble_member: str,
    domain: str,
    collection: str,
    base_dir: Path,
) -> xr.Dataset:
    source_metadata = CanariLESprintVariableAdapter(
        frequency=frequency,
        ensemble_member=ensemble_member,
        variable=src_variable,
        year=year,
    )

    ds = source_metadata.open().load()

    return ds


def combine_source_variables(sources: dict[str, xr.Dataset]) -> xr.Dataset:
    logger.info(f"Combining source variables...")

    return xr.combine_by_coords(
        sources.values(),
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )


def open_source_variables(
    src_configs: set[SourceConfig],
    year: int,
    ensemble_member: str,
    base_dir: Path,
) -> xr.Dataset:
    sources = {}
    for src_config in src_configs:

        src_type = src_config.type

        if src_type == "moose":
            source_open_strategy = open_moose_source_variable
        elif src_type == "local":
            source_open_strategy = open_local_source_variable
        elif src_type == "canari-le-sprint":
            source_open_strategy = open_canari_le_sprint_source_variable
        else:
            raise RuntimeError(f"Unknown source type {src_type}")

        collection = src_config.collection
        resolution = get_resolution(src_config)
        frequency = src_config.frequency
        scenario = "rcp85"
        domain = get_source_domain(src_config)

        sources[src_config.variable] = source_open_strategy(
            src_config.variable,
            year,
            frequency,
            scenario,
            resolution,
            ensemble_member,
            domain,
            collection,
            base_dir,
        )

    logger.info(f"Combining {src_configs}...")
    ds = combine_source_variables(sources).assign_attrs(
        {
            "domain": domain,
            "resolution": resolution,
            "frequency": frequency,
        }
    )

    return ds


def _process(
    ds: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    for job_spec in config["spec"]:
        if job_spec["action"] in [
            "sum",
            "diff",
            "query",
            "shift_lon_break",
            "vorticity",
            "coarsen",
            "select-subdomain",
            "resample",
            "rename",
            "drop-variables",
        ]:
            typer.echo(f"Doing {job_spec['action']}...")
            ds = get_action(job_spec["action"])(**job_spec.get("parameters", {}))(ds)
        elif job_spec["action"] == "regrid_to_target":
            # this assumes mapping to a target grid of higher resolution than resolution of the data
            ds = get_action(job_spec["action"])(
                variables=[config["variable"]], **job_spec.get("parameters", {})
            )(ds)
        else:
            raise RuntimeError(f"Unknown action {job_spec['action']}")

    # assign any attributes from config file
    ds[config["variable"]] = ds[config["variable"]].assign_attrs(config["attrs"])

    return ds


def _validate(ds: xr.Dataset, config: dict) -> None:
    if ds.attrs["frequency"] == "day":
        # there should be 360 days in the dataset
        assert len(ds.time) == 360

    # there should be no missing values in this dataset
    assert ds[config["variable"]].isnull().sum().values.item() == 0


def _save(ds: xr.Dataset, config: dict, path: str, year: int) -> None:
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
    config_paths: list[Path] = typer.Option(...),
    thetas: list[int] = None,
    scenario="rcp85",
    ensemble_member: str = typer.Option(...),
    year: int = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    target_resolution: str = None,
    input_base_dir: Path = None,
    output_base_dir: Path = None,
    validate: bool = True,
):
    """
    Create a variable file in project form from source data
    """

    configs = [
        load_config(
            config_path,
            scale_factor=scale_factor,
            domain=domain.value,
            theta=theta,
            target_resolution=target_resolution,
        )
        for config_path in config_paths
        for theta in (thetas or [None])
    ]

    src_configs = {
        SourceConfig(
            type=config["sources"]["type"],
            collection=config["sources"]["collection"],
            frequency=config["sources"]["frequency"],
            variable=var_configs["name"],
        )
        for config in configs
        for var_configs in config["sources"]["variables"]
    }
    src_type = {src_config.type for src_config in src_configs}
    # TODO: support creating variables from multiple source types
    assert len(src_type) == 1, "All variable configs must have the same source type"
    src_type = src_type.pop()

    if input_base_dir is None:
        if src_type == "moose":
            input_base_dir = RAW_MOOSE_VARIABLES_PATH
        elif src_type == "local":
            input_base_dir = DERIVED_VARIABLES_PATH
        elif src_type == "canari-le-sprint":
            input_base_dir = None
        else:
            raise RuntimeError(f"Unknown source type {src_type}")

    if output_base_dir is None:
        output_base_dir = DERIVED_VARIABLES_PATH

    src_ds = open_source_variables(
        src_configs,
        year,
        ensemble_member,
        input_base_dir,
    )
    for config in configs:
        logger.info(f"Processing {config['variable']}...")
        ds = _process(
            src_ds,
            config,
        )
        # # remove pressure related dims and encoding data that we don't need
        # ds = remove_pressure(ds)

        if validate:
            _validate(ds, config)

        output_metadata = VariableMetadata(
            output_base_dir,
            frequency=ds.attrs["frequency"],
            domain=ds.attrs["domain"],
            resolution=ds.attrs["resolution"],
            scenario=scenario,
            ensemble_member=ensemble_member,
            variable=config["variable"],
            collection=config["sources"]["collection"],
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
                                f"{DERIVED_VARIABLES_PATH}",
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
