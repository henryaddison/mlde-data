import gc
import logging
from mlde_utils import VariableMetadata
from pathlib import Path
import xarray as xr

from .random_split import RandomSplit
from .random_season_split import RandomSeasonSplit
from .season_stratified_intensity_split import SeasonStratifiedIntensitySplit

logger = logging.getLogger(__name__)


def create(config: dict, input_base_dir: Path):
    """
    Create a dataset
    """
    single_em_datasets = []

    for em in config["ensemble_members"]:

        single_em_ds = _combine_variables(em, config, input_base_dir)

        single_em_ds = single_em_ds.assign_coords(
            season=(("time"), (single_em_ds["time.month"].values % 12 // 3))
        )

        single_em_datasets.append(single_em_ds)

        del single_em_ds
        gc.collect()
        logger.debug(f"Gathered data for {em}")

    multi_em_ds = xr.concat(
        single_em_datasets,
        dim="ensemble_member",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="exact",
        data_vars="minimal",
    )
    del single_em_datasets
    gc.collect()

    split_sets = _split(multi_em_ds, **config["split"])

    return split_sets


def _combine_variables(em: str, config: dict, input_base_dir: Path):
    """
    Combine predictor and predictand variables for a given ensemble into a single dataset
    """

    common_var_params = {k: config[k] for k in ["domain", "scenario", "frequency"]}

    variable_datasets = []
    for var_type in ["predictors", "predictands"]:
        var_type_config = config[var_type]
        for predictor_var_name in var_type_config["variables"]:
            dsmeta = VariableMetadata(
                input_base_dir / "moose",
                ensemble_member=em,
                variable=predictor_var_name,
                resolution=var_type_config["resolution"],
                **common_var_params,
            )

            variable_ds = xr.open_mfdataset(
                dsmeta.existing_filepaths(),
                data_vars="minimal",
                combine="by_coords",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            )
            variable_ds[dsmeta.variable] = variable_ds[dsmeta.variable].expand_dims(
                dict(ensemble_member=[em])
            )
            if var_type == "predictands":
                variable_ds = variable_ds.rename(
                    {dsmeta.variable: f"target_{dsmeta.variable}"}
                )

            variable_datasets.append(variable_ds)

    single_em_ds = xr.combine_by_coords(
        variable_datasets,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="exact",
        data_vars="minimal",
    )

    return single_em_ds


def _split(ds: xr.Dataset, scheme: str, val_prop: float, test_prop: float, seed: int):
    """
    Split data into train, validation and test subsets
    """
    if scheme == "ssi":
        splitter = SeasonStratifiedIntensitySplit(
            val_prop=val_prop,
            test_prop=test_prop,
            seed=seed,
        )
    elif scheme == "random":
        splitter = RandomSplit(
            val_prop=val_prop,
            test_prop=test_prop,
            seed=seed,
        )
    elif scheme == "random-season":
        splitter = RandomSeasonSplit(
            val_prop=val_prop,
            test_prop=test_prop,
            seed=seed,
        )
    else:
        raise RuntimeError(f"Unknown split scheme {scheme}")
    logger.info(f"Splitting data...")
    return splitter.run(ds)
