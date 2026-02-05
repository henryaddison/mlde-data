from collections import defaultdict
import gc
import logging
from mlde_utils import VariableMetadata
from pathlib import Path
import re
import xarray as xr

from mlde_utils import DatasetMetadata

from .random_split import RandomSplit
from .random_season_split import RandomSeasonSplit
from .season_stratified_intensity_split import SeasonStratifiedIntensitySplit

logger = logging.getLogger(__name__)


def create(config: dict, input_base_dir: Path) -> dict:
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


def validate(dataset: str) -> defaultdict:
    """
    Validate a dataset
    """
    splits = ["train", "val", "test"]

    bad_splits = defaultdict(set)

    try:
        ds_config = DatasetMetadata(dataset).config()
    except FileNotFoundError:
        bad_splits["no config"].update(splits)
        return bad_splits

    for split in splits:
        split_path = DatasetMetadata(dataset).split_path(split)
        try:
            ds = xr.open_dataset(split_path)
        except FileNotFoundError:
            bad_splits["no file"].add(split)
            continue

        # check dims
        if not check_dims(ds, dataset, split, ds_config):
            bad_splits["bad dimensions"].add(split)

        # check grid and time
        if not check_grid_vars(ds, dataset, split, ds_config):
            bad_splits["bad grid vars"].add(split)
        if not check_time_bnds(ds, dataset, split, ds_config):
            bad_splits["bad time_bnds"].add(split)
        if not check_time_encoding(ds, dataset, split, ds_config):
            bad_splits["bad time encodings"].add(split)

        # check shape
        if not check_shape(ds, dataset, split, ds_config):
            bad_splits["bad shape"].add(split)

        # check for forecast related metadata (should have been stripped)
        if not check_forecast_encoding(ds, dataset, split, ds_config):
            bad_splits["forecast_encoding"].add(split)
        if not check_forecast_variables(ds, dataset, split, ds_config):
            bad_splits["forecast_vars"].add(split)

        # check for pressure related metadata (should have been stripped)
        if not check_pressure_encoding(ds, dataset, split, ds_config):
            bad_splits["pressure_encoding"].add(split)
        if not check_forecast_variables(ds, dataset, split, ds_config):
            bad_splits["pressure_vars"].add(split)

        # check for NaNs
        if not check_nans(ds, dataset, split, ds_config):
            bad_splits["NaNs"].add(split)

    return bad_splits


def check_grid_mapping(ds, dataset, split, ds_config):
    target_vars = list(
        map(lambda v: f"target_{v}", ds_config["predictands"]["variables"])
    )
    input_vars = ds_config["predictors"]["variables"]

    grid_mappings = set(
        map(lambda v: ds[v].attrs["grid_mapping"], target_vars + input_vars)
    )

    if len(grid_mappings) != 1:
        return False
    if grid_mappings[0] not in ["rotated_latitude_longitude", "latitude_longitude"]:
        return False

    return True


def check_dims(ds, dataset, split, ds_config):
    example_var = f"target_{ds_config['predictands']['variables'][0]}"
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[example_var].dims) == [
            "ensemble_member",
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[example_var].dims) == [
            "ensemble_member",
            "time",
            "latitude",
            "longitude",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_shape(ds, dataset, split, ds_config):
    ems = ds_config["ensemble_members"]
    example_var = f"target_{ds_config['predictands']['variables'][0]}"
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        size = 64
    elif grid_mapping == "latitude_longitude":
        size = 9
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")

    if split == "train":
        expected_shape = (len(ems), 360 * 14 * 3, size, size)
    else:
        expected_shape = (len(ems), 360 * 3 * 3, size, size)
    return ds[example_var].shape == expected_shape


def check_grid_vars(ds, dataset, split, ds_config):
    example_var = f"target_{ds_config['predictands']['variables'][0]}"
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    meta_vars = [
        grid_mapping,
    ]
    if grid_mapping == "rotated_latitude_longitude":
        meta_vars.extend(["grid_latitude_bnds", "grid_longitude_bnds"])

    return all(
        [
            ("ensemble_member" not in ds[mvar].dims) and ("time" not in ds[mvar].dims)
            for mvar in meta_vars
        ]
    )


def check_time_bnds(ds, dataset, split, ds_config):
    return "ensemble_member" not in ds["time_bnds"].dims


def check_forecast_encoding(ds, dataset, split, ds_config):
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match(
                "(realization|forecast_period|forecast_reference_time) ?",
                ds[v].encoding["coordinates"],
            )
            is not None
        ):
            return False
    return True


def check_forecast_variables(ds, dataset, split, ds_config):
    for v in ds.variables:
        if v in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "forecast_period_bnds",
        ]:
            return False
    return True


def check_pressure_encoding(ds, dataset, split, ds_config):
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
        ):
            return False
    return True


def check_pressure_variables(ds, dataset, split, ds_config):
    for v in ds.variables:
        if v in ["pressure"]:
            return False
    return True


def check_nans(ds, dataset, split, ds_config):
    for v in ds.variables:
        nan_count = ds[v].isnull().sum().values.item()
        if nan_count > 0:
            return False
    return True


def check_time_encoding(ds, dataset, split, ds_config):
    for enc in [ds.time.encoding, ds.time_bnds.encoding]:
        if enc["units"] != "hours since 1970-01-01":
            return False
        if enc["calendar"] != "360_day":
            return False

    return True


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
                input_base_dir,
                ensemble_member=em,
                variable=predictor_var_name,
                resolution=var_type_config["resolution"],
                collection=var_type_config["collection"],
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
