from collections import defaultdict
import gc
import glob
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import yaml

import typer
import xarray as xr

from mlde_utils import (
    VariableMetadata,
    TIME_PERIODS,
    dataset_path,
    dataset_config_path,
    dataset_config,
    dataset_split_path,
)
from ..dataset import (
    RandomSplit,
    RandomSeasonSplit,
    SeasonStratifiedIntensitySplit,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def create(
    config: Path,
    input_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    output_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
):
    """
    Create a dataset
    """
    config_name = config.stem
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    split_scheme = config["split"]["scheme"]
    val_prop: float = config["split"]["val_prop"]
    test_prop: float = config["split"]["test_prop"]
    split_seed: int = config["split"]["seed"]

    single_em_datasets = []

    for em in config["ensemble_members"]:

        predictand_var_params = {
            k: config[k] for k in ["domain", "scenario", "frequency"]
        }
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
                VariableMetadata(
                    input_base_dir / "moose", ensemble_member=em, **var_params
                )
            )

        example_predictor_filepath = predictors_meta[0].existing_filepaths()[0]
        time_encoding = xr.open_dataset(example_predictor_filepath).time_bnds.encoding

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

        single_em_ds = single_em_ds.assign_coords(
            season=(("time"), (single_em_ds["time.month"].values % 12 // 3))
        )

        single_em_datasets.append(single_em_ds)

        del predictor_datasets, predictand_ds, single_em_ds
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

    if split_scheme == "ssi":
        splitter = SeasonStratifiedIntensitySplit(
            val_prop=val_prop,
            test_prop=test_prop,
            time_encoding=time_encoding,
            seed=split_seed,
        )
    elif split_scheme == "random":
        splitter = RandomSplit(
            val_prop=val_prop,
            test_prop=test_prop,
            time_encoding=time_encoding,
            seed=split_seed,
        )
    elif split_scheme == "random-season":
        splitter = RandomSeasonSplit(
            val_prop=val_prop,
            test_prop=test_prop,
            time_encoding=time_encoding,
            seed=split_seed,
        )
    else:
        raise RuntimeError(f"Unknown split scheme {split_scheme}")
    logger.info(f"Splitting data...")
    split_sets = splitter.run(multi_em_ds)

    output_dir = dataset_path(config_name, base_dir=output_base_dir)

    os.makedirs(output_dir, exist_ok=False)

    logger.info(f"Saving data to {output_dir}...")
    with open(dataset_config_path(config_name, base_dir=output_base_dir), "w") as f:
        yaml.dump(config, f)
    for split_name, split_ds in split_sets.items():
        for varname in split_ds.data_vars:
            split_ds[varname].encoding.update(zlib=True, complevel=5)
        split_ds.to_netcdf(os.path.join(output_dir, f"{split_name}.nc"))
        logger.info(f"{split_name} done")


def check_dims(ds, dataset, split, ds_config):
    var = "target_pr"
    grid_mapping = ds[var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[var].dims) == [
            "ensemble_member",
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[var].dims) == [
            "ensemble_member",
            "time",
            "latitude",
            "longitude",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_shape(ds, dataset, split, ds_config):
    ems = ds_config["ensemble_members"]
    grid_mapping = ds["target_pr"].attrs["grid_mapping"]
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
    return ds["target_pr"].shape == expected_shape


def check_grid_vars(ds, dataset, split, ds_config):
    grid_mapping = ds["target_pr"].attrs["grid_mapping"]
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


def report_issues(dataset, bad_splits):
    for reason, error_splits in bad_splits.items():
        if len(error_splits) > 0:
            print(f"Failed '{reason}': {dataset} for {error_splits}")


@app.command()
def validate(dataset_name: str = typer.Argument("all")):
    datasets = [
        "bham_60km-4x_1em_vort850_eqvt_random-season",
        "bham_60km-4x_1em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_60km-4x_12em_linpr_eqvt_random-season",
        "bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_60km-4x_12em_psl-temp4th-vort4th_eqvt_random-season",
        "bham_60km-4x_12em_vort4th_eqvt_random-season",
        "bham_60km-4x_12em_vort850_eqvt_random-season",
        "bham_60km-60km_1em_rawpr_eqvt_random-season",
        "bham_60km-60km_12em_rawpr_eqvt_random-season",
        "bham_gcmx-4x_1em_vort850_eqvt_random-season",
        "bham_gcmx-4x_1em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_linpr_eqvt_random-season",
        "bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_vort850_eqvt_random-season",
        "bham_gcmx-60km_1em_pr_eqvt_random-season",
        "bham_gcmx-60km_12em_pr_eqvt_random-season",
    ]

    splits = ["train", "val", "test"]

    for dataset in datasets:
        if (dataset_name != "all") and (dataset_name != dataset):
            continue
        bad_splits = defaultdict(set)

        try:
            ds_config = dataset_config(dataset)
        except FileNotFoundError:
            bad_splits["no config"].update(splits)
            report_issues(dataset, bad_splits)
            continue

        for split in splits:
            sys.stdout.write("\033[K")
            print(f"Checking {split} of {dataset}", end="\r")

            split_path = dataset_split_path(dataset, split)
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

        # report findings
        report_issues(dataset, bad_splits)


@app.command()
def random_subset(
    src_dataset: str,
    dest_dataset: str,
    pc: int = 50,
    split: str = "train",
    seed: int = 42,
):
    src_dataset_dir = dataset_path(src_dataset)
    dest_dataset_dir = dataset_path(dest_dataset)

    logger.info(f"Copying {src_dataset_dir} to {dest_dataset_dir}...")
    # os.makedirs(dest_dataset_dir, exist_ok=True)
    shutil.copytree(src_dataset_dir, dest_dataset_dir)

    new_split_filepath = dest_dataset_dir / f"{split}.nc"

    logger.info(f"Subsetting {new_split_filepath}")
    original_split = xr.open_dataset(new_split_filepath)
    new_size = int(len(original_split["time"]) * pc / 100.0)
    rng = np.random.default_rng(seed=seed)
    time_subset = rng.choice(
        original_split["time"].values, size=new_size, replace=False
    )
    new_split = original_split.sel(time=time_subset).load()
    original_split.close()
    new_split.to_netcdf(new_split_filepath)


@app.command()
def random_subset_split(
    dataset: str,
    split: str,
    pc: int = 50,
    new_split: str = None,
    seed: int = 42,
):
    dataset_dir = dataset_path(dataset)

    orig_split_filepath = dataset_dir / f"{split}.nc"
    if new_split is None:
        new_split = f"{split}-{pc}pc"
    new_split_filepath = dataset_dir / f"{new_split}.nc"

    logger.info(f"Subsetting {orig_split_filepath}")
    original_split = xr.open_dataset(orig_split_filepath)
    new_size = int(len(original_split["time"]) * pc / 100.0)
    rng = np.random.default_rng(seed=seed)
    time_subset = rng.choice(
        original_split["time"].values, size=new_size, replace=False
    )
    time_subset.sort()
    new_split = original_split.sel(time=time_subset).load()
    original_split.close()
    new_split.to_netcdf(new_split_filepath)


@app.command()
def filter(
    dataset: str,
    time_period: str,
    base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
):

    input_dir = dataset_path(dataset, base_dir=base_dir)
    config = dataset_config(dataset, base_dir=base_dir)

    config_filters = config.get("filters", list())
    config_filters.append({"time_period": time_period})
    config["filters"] = config_filters

    new_dataset = f"{dataset}-{time_period}"
    output_dir = dataset_path(new_dataset, base_dir=base_dir)
    os.makedirs(output_dir, exist_ok=False)
    for split_filepath in glob.glob(os.path.join(input_dir, "*.nc")):
        split_file = os.path.basename(split_filepath)
        logger.info(f"Filtering {split_file} to {output_dir}")
        split_ds = xr.open_dataset(split_filepath)
        output_filepath = os.path.join(output_dir, split_file)
        split_ds.sel(time=slice(*TIME_PERIODS[time_period])).to_netcdf(output_filepath)

    with open(dataset_config_path(new_dataset, base_dir=base_dir), "w") as f:
        yaml.dump(config, f)


@app.command()
def quantile(
    dataset: str,
    p: float,
    variable: str = "target_pr",
    base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    split: str = "train",
):
    input_dir = dataset_path(dataset, base_dir=base_dir)

    split_ds = xr.open_dataset(os.path.join(input_dir, f"{split}.nc"))
    Q_p = split_ds[variable].quantile(p)
    typer.echo(Q_p.values.item())
