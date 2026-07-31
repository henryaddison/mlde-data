import cf_xarray  # noqa: F401
import gc
import logging
from mlde_utils import VariableMetadata
from pathlib import Path
import xarray as xr

from .preset_split import PresetSplit
from .random_split import RandomSplit
from .random_season_split import RandomSeasonSplit

logger = logging.getLogger(__name__)


def _calculate_statistics(
    split_ds: xr.Dataset, variables: list[str], time_aggregation_factors: list[int]
) -> xr.Dataset:
    """
    Calculate statistics for each variable in the dataset

    Used for transforming the data later (e.g. standardization in ML pipeline)
    """

    return xr.combine_nested(
        [
            [
                xr.merge(
                    [
                        split_ds[var]
                        .coarsen(time=time_factor)
                        .sum()
                        .count()
                        .rename("count"),
                        split_ds[var]
                        .coarsen(time=time_factor)
                        .sum()
                        .mean()
                        .rename("mean"),
                        split_ds[var]
                        .coarsen(time=time_factor)
                        .sum()
                        .std()
                        .rename("std"),
                        split_ds[var]
                        .coarsen(time=time_factor)
                        .sum()
                        .max()
                        .rename("max"),
                        split_ds[var]
                        .coarsen(time=time_factor)
                        .sum()
                        .min()
                        .rename("min"),
                    ]
                ).expand_dims(
                    {"variable": [var], "time_aggregation_factor": [time_factor]}
                )
                for var in variables
            ]
            for time_factor in time_aggregation_factors
        ],
        concat_dim=["time_aggregation_factor", "variable"],
        data_vars="minimal",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="exact",
    )


def create(config: dict, input_base_dir: Path) -> dict:
    """
    Create a dataset
    """
    scenario = config["scenario"]

    var_type_datasets = {}
    var_type_statistics = {}
    split_sets = None
    for var_type in ["predictands", "predictors"]:
        logger.info(f"Processing {var_type}...")

        var_type_datasets[var_type] = {}
        var_type_statistics[var_type] = {}
        var_type_config = config[var_type]
        single_var_datasets = []
        for var_name in var_type_config["variables"]:
            logger.info(f"Processing {var_name}...")
            single_em_var_datasets = []
            for em in config["ensemble_members"]:
                single_em_var_datasets.append(
                    _single_variable(
                        em,
                        var_name,
                        input_base_dir=input_base_dir,
                        resolution=var_type_config["resolution"],
                        collection=var_type_config["collection"],
                        frequency=var_type_config["frequency"],
                        domain=var_type_config["domain"],
                        scenario=scenario,
                    )
                )
            logger.info(f"Combining ensemble members for {var_name}...")
            multi_em_ds = xr.concat(
                single_em_var_datasets,
                dim="ensemble_member",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
                join="exact",
                data_vars="minimal",
            )
            single_var_datasets.append(multi_em_ds)

            del single_em_var_datasets
            gc.collect()

            if split_sets is None:
                logger.info(f"Generating times for split sets...")
                split_sets = _split(multi_em_ds["time"], **config["split"])
        logger.info(f"Combining variables for {var_type}...")
        var_type_ds = xr.combine_by_coords(
            single_var_datasets,
            compat="no_conflicts",
            combine_attrs="drop_conflicts",
            join="exact",
            data_vars="minimal",
        )

        logger.info(f"Splitting data for {var_type}...")
        for split, split_times in split_sets.items():
            split_ds = var_type_ds.sel(
                time=var_type_ds["time"].dt.floor("D").isin(split_times)
            )
            var_type_datasets[var_type][split] = split_ds

            var_type_statistics[var_type][split] = _calculate_statistics(
                split_ds,
                var_type_config["variables"],
                **config[var_type].get("stats", {"time_aggregation_factors": [1]}),
            )

    return var_type_datasets, var_type_statistics


def _single_variable(
    em: str, var_name, input_base_dir: Path, **var_config: dict
) -> xr.Dataset:
    """
    Combine files for a given ensemble member and variable into single xarray.Dataset
    """

    dsmeta = VariableMetadata(
        input_base_dir, ensemble_member=em, variable=var_name, **var_config
    )

    # basically just concat along time dimension but want to preserve order
    variable_ds = xr.open_mfdataset(
        dsmeta.existing_filepaths(),
        data_vars="minimal",
        combine="by_coords",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="outer",
    )
    variable_ds[dsmeta.variable] = variable_ds[dsmeta.variable].expand_dims(
        dict(ensemble_member=[em])
    )
    return variable_ds


def _split(
    time_da: xr.DataArray,
    scheme: str,
    **splitter_kwargs: dict,
):
    """
    Split data into train, validation and test subsets
    """
    if scheme == "random":
        splitter = RandomSplit
    elif scheme == "random-season":
        splitter = RandomSeasonSplit
    elif scheme == "preset":
        splitter = PresetSplit
    else:
        raise RuntimeError(f"Unknown split scheme {scheme}")

    return splitter(**splitter_kwargs).run(time_da)
