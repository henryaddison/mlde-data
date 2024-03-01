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
    predictand_var_params = {k: config[k] for k in ["domain", "scenario", "frequency"]}
    predictand_var_params.update(
        {
            "variable": config["predictand"]["variable"],
            "resolution": config["predictand"]["resolution"],
        }
    )
    predictand_meta = VariableMetadata(
        input_base_dir / "moose", ensemble_member=em, **predictand_var_params
    )

    predictors_meta = []
    for predictor_var_config in config["predictors"]:
        var_params = {
            k: config[k]
            for k in [
                "domain",
                "scenario",
                "frequency",
                "resolution",
            ]
        }
        var_params.update({k: predictor_var_config[k] for k in ["variable"]})
        predictors_meta.append(
            VariableMetadata(input_base_dir / "moose", ensemble_member=em, **var_params)
        )

    predictor_datasets = []
    for dsmeta in predictors_meta:
        predictor_ds = xr.open_mfdataset(
            dsmeta.existing_filepaths(),
            data_vars="minimal",
            combine="by_coords",
            compat="no_conflicts",
            combine_attrs="drop_conflicts",
        )
        predictor_ds[dsmeta.variable] = predictor_ds[dsmeta.variable].expand_dims(
            dict(ensemble_member=[em])
        )

        predictor_datasets.append(predictor_ds)

    predictand_ds = xr.open_mfdataset(
        predictand_meta.existing_filepaths(),
        data_vars="minimal",
        combine="by_coords",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    )
    predictand_ds[predictand_meta.variable] = predictand_ds[
        predictand_meta.variable
    ].expand_dims(dict(ensemble_member=[em]))
    predictand_ds = predictand_ds.rename(
        {predictand_meta.variable: f"target_{predictand_meta.variable}"}
    )

    single_em_ds = xr.combine_by_coords(
        [*predictor_datasets, predictand_ds],
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        join="exact",
        data_vars="minimal",
    )

    # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
    example_predictor_filepath = predictors_meta[0].existing_filepaths()[0]
    example_ds = xr.open_dataset(example_predictor_filepath)
    single_em_ds.time.encoding.update(example_ds.time.encoding)
    single_em_ds.time_bnds.encoding.update(example_ds.time_bnds.encoding)

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
