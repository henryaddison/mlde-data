from dataclasses import dataclass
import logging
from mlde_utils import VariableMetadata
from pathlib import Path
from string import Template
import xarray as xr
import yaml
from mlde_data.actions.actions_registry import get_action
from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter
from mlde_data.ceda_variable_adapter import CedaVariableAdapter
from mlde_data.moose_extract_variable_adapter import MooseExtractVariableAdapter
from mlde_data.moose import (
    remove_forecast,
    remove_pressure,
)
from mlde_data.options import CollectionOption

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceVariableConfig:
    src_type: str
    collection: str
    frequency: str
    variable: str
    resolution: str
    domain: str = None

    def __post_init__(self):
        if self.src_type == "moose" or self.src_type == "ceda":
            if self.collection == CollectionOption.cpm:
                if self.domain is None:
                    object.__setattr__(self, "domain", "uk")
            elif self.collection == CollectionOption.gcm:
                if self.domain is None:
                    object.__setattr__(self, "domain", "global")
            else:
                raise f"Unknown collection {self.collection}"
        elif self.src_type == "local":
            # assume local sourced data is pre-processed so resolution and domain must be specified
            assert (
                self.domain is not None
            ), "domain must be specified for local source variable"
        elif self.src_type == "canari-le-sprint":
            if self.domain is None:
                object.__setattr__(self, "domain", "global")
        else:
            raise RuntimeError(f"Unknown souce type {self.src_type}")


def load_config(
    config_path: Path,
    scale_factor: str,
    domain: str,
    theta: int = None,
    target_resolution: str = None,
):
    """Load configuration for a creating a variable from a YAML file."""

    with open(config_path, "r") as config_template:
        d = {
            "scale_factor": scale_factor,
            "domain": domain,
            "theta": theta,
            "target_resolution": target_resolution,
        }
        src = Template(config_template.read())
        result = src.substitute(d)
        config = yaml.safe_load(result)

    config["sources"] = {
        SourceVariableConfig(
            src_type=config["sources"]["type"],
            collection=config["sources"]["collection"],
            frequency=config["sources"]["frequency"],
            resolution=config["sources"]["resolution"],
            variable=var_configs["name"],
        )
        for var_configs in config["sources"]["variables"]
    }

    return config


def open_source_variables(
    src_configs: set[SourceVariableConfig],
    year: int,
    ensemble_member: str,
    base_dir: Path,
) -> xr.Dataset:
    sources = {}
    for src_config in src_configs:

        src_type = src_config.src_type

        if src_type == "moose":
            source_open_strategy = _open_moose_extract_source_variable
        elif src_type == "ceda":
            source_open_strategy = _open_ceda_source_variable
        elif src_type == "local":
            source_open_strategy = _open_local_source_variable
        elif src_type == "canari-le-sprint":
            source_open_strategy = _open_canari_le_sprint_source_variable
        else:
            raise RuntimeError(f"Unknown source type {src_type}")

        collection = src_config.collection
        resolution = src_config.resolution
        frequency = src_config.frequency
        scenario = "rcp85"
        domain = src_config.domain

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
    ds = _combine_source_variables(sources).assign_attrs(
        {
            "domain": domain,
            "resolution": resolution,
            "frequency": frequency,
        }
    )

    return ds


def build(src_ds: xr.Dataset, config: dict) -> xr.Dataset:
    """
    Construct a new variable as an xarray Dataset based on config from a source dataset
    """
    ds = _process(
        src_ds,
        config,
    )
    # remove pressure related dims and encoding data that we don't need
    ds = remove_pressure(ds)

    return ds


def _process(
    ds: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    for job_spec in config["spec"]:
        logger.info(f"Doing {job_spec['action']}...")
        if job_spec["action"] == "regrid_to_target":
            # this assumes mapping to a target grid of higher resolution than resolution of the data
            action = get_action(job_spec["action"])(
                variables=[config["variable"]], **job_spec.get("parameters", {})
            )
        else:
            action = get_action(job_spec["action"])(**job_spec.get("parameters", {}))
        ds = action(ds)

    # assign any attributes from config file
    if "attrs" in config:
        ds[config["variable"]] = ds[config["variable"]].assign_attrs(config["attrs"])

    return ds


def _open_local_source_variable(
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


def _open_moose_extract_source_variable(
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
    source_metadata = MooseExtractVariableAdapter(
        frequency=frequency,
        ensemble_member=ensemble_member,
        variable=src_variable,
        year=year,
        scenario=scenario,
        resolution=resolution,
        domain=domain,
        collection=collection,
        base_dir=base_dir,
    )

    ds = source_metadata.open()

    # remove forecast related coords that we don't need
    ds = remove_forecast(ds)

    return ds


def _open_canari_le_sprint_source_variable(
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


def _open_ceda_source_variable(
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
    logger.info(f"Opening {src_variable} from CEDA...")
    source_metadata = CedaVariableAdapter(
        frequency=frequency,
        ensemble_member=ensemble_member,
        variable=src_variable,
        year=year,
        scenario=scenario,
        resolution=resolution,
        domain=domain,
        collection=collection,
        base_dir=base_dir,
    )

    ds = source_metadata.open()

    return ds


def _combine_source_variables(sources: dict[str, xr.Dataset]) -> xr.Dataset:
    logger.info(f"Combining source variables...")

    return xr.combine_by_coords(
        sources.values(),
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )
