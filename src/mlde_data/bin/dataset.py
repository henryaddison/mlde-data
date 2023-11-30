from collections import defaultdict
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

from mlde_utils import VariableMetadata, TIME_PERIODS
from ..dataset import (
    RandomSplit,
    RandomSeasonSplit,
    SeasonStratifiedIntensitySplit,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def create(
    config: Path,
    input_base_dir: Path = typer.Argument(..., envvar="MOOSE_DERIVED_DATA"),
    output_base_dir: Path = typer.Argument(..., envvar="MOOSE_DERIVED_DATA"),
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

    combined_datasets = []

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
            input_base_dir, ensemble_member=em, **predictand_var_params
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
                VariableMetadata(input_base_dir, ensemble_member=em, **var_params)
            )

        example_predictor_filepath = predictors_meta[0].existing_filepaths()[0]
        time_encoding = xr.open_dataset(example_predictor_filepath).time_bnds.encoding

        predictor_datasets = [
            xr.open_mfdataset(dsmeta.existing_filepaths()) for dsmeta in predictors_meta
        ]
        predictand_dataset = xr.open_mfdataset(
            predictand_meta.existing_filepaths()
        ).rename({predictand_meta.variable: f"target_{predictand_meta.variable}"})

        combined_dataset = xr.combine_by_coords(
            [*predictor_datasets, predictand_dataset],
            compat="override",
            combine_attrs="drop_conflicts",
            coords="all",
            join="inner",
            data_vars="all",
        )
        combined_dataset = combined_dataset.assign_coords(
            season=(("time"), (combined_dataset["time.month"].values % 12 // 3))
        )

        combined_dataset = combined_dataset.expand_dims(dict(ensemble_member=[em]))
        combined_datasets.append(combined_dataset)

    combined_dataset = xr.concat(combined_datasets, dim="ensemble_member")

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

    split_sets = splitter.run(combined_dataset)

    output_dir = os.path.join(output_base_dir, "nc-datasets", config_name)

    os.makedirs(output_dir, exist_ok=False)

    logger.info(f"Saving data to {output_dir}")
    with open(os.path.join(output_dir, "ds-config.yml"), "w") as f:
        yaml.dump(config, f)
    for split_name, split_ds in split_sets.items():
        split_ds.to_netcdf(os.path.join(output_dir, f"{split_name}.nc"))


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
    if split == "train":
        expected_shape = (len(ems), 360 * 14 * 3, 64, 64)
    elif "_eqvt_" in dataset:
        expected_shape = (len(ems), 360 * 3 * 3, 64, 64)
    elif split == "test":
        expected_shape = (len(ems), 360 * 2 * 3, 64, 64)
    else:
        expected_shape = (len(ems), 360 * 4 * 3, 64, 64)
    return ds["target_pr"].shape == expected_shape


@app.command()
def validate(dataset_name: str = typer.Argument("all")):
    datasets = [
        "bham_60km-4x_12em_linpr_eqvt_random-season",
        "bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-historic",
        "bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-present",
        "bham_60km-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-future",
        "bham_60km-4x_12em_psl-temp4th-vort4th_eqvt_random-season",
        "bham_60km-4x_12em_psl-temp4th-vort4th_eqvt_random-season-historic",
        "bham_60km-4x_12em_psl-temp4th-vort4th_eqvt_random-season-present",
        "bham_60km-4x_12em_psl-temp4th-vort4th_eqvt_random-season-future",
        "bham_60km-4x_1em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_60km-60km_12em_rawpr_eqvt_random-season",
        "bham_60km-60km_rawpr_eqvt_random-season",
        "bham_gcmx-4x_12em_linpr_eqvt_random-season",
        "bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-historic",
        "bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-present",
        "bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-future",
        "bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season-historic",
        "bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season-present",
        "bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season-future",
        "bham_gcmx-4x_1em_psl-sphum4th-temp4th-vort4th_eqvt_random-season",
        "bham_gcmx-4x_1em_psl-sphum4th-temp4th-vort4th_eqvt_random-season-historic",
        # "bham_gcmx-60km_12em_pr_eqvt_random-season",
        "bham_gcmx-60km_pr_eqvt_random-season",
    ]

    splits = ["train", "val", "test"]

    for dataset in datasets:
        if (dataset_name != "all") and (dataset_name != dataset):
            continue
        bad_splits = defaultdict(set)
        for split in splits:
            sys.stdout.write("\033[K")
            print(f"Checking {split} of {dataset}", end="\r")
            dataset_path = os.path.join(
                os.getenv("MOOSE_DERIVED_DATA"), "nc-datasets", dataset
            )
            ds_config_path = os.path.join(dataset_path, "ds-config.yml")
            with open(ds_config_path, "r") as f:
                ds_config = yaml.safe_load(f)
            split_path = os.path.join(dataset_path, f"{split}.nc")
            try:
                ds = xr.open_dataset(split_path)
            except FileNotFoundError:
                bad_splits["no file"].add(split)
                continue

            # check dims
            if not check_dims(ds, dataset, split, ds_config):
                bad_splits["bad dimensions"].add(split)

            # check shape
            if not check_shape(ds, dataset, split, ds_config):
                bad_splits["bad shape"].add(split)

            # check for forecast related metadata (should have been stripped)
            for v in ds.variables:
                if "coordinates" in ds[v].encoding and (
                    re.match(
                        "(realization|forecast_period|forecast_reference_time) ?",
                        ds[v].encoding["coordinates"],
                    )
                    is not None
                ):
                    bad_splits["forecast_encoding"].add(split)
                if v in [
                    "forecast_period",
                    "forecast_reference_time",
                    "realization",
                    "forecast_period_bnds",
                ]:
                    bad_splits["forecast_vars"].add(split)

            # check for pressure related metadata (should have been stripped)
            for v in ds.variables:
                if "coordinates" in ds[v].encoding and (
                    re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
                ):
                    bad_splits["pressure_encoding"].add(split)
                if v in ["pressure"]:
                    bad_splits["pressure_vars"].add(split)

            # check for NaNs
            for v in ds.variables:
                nan_count = ds[v].isnull().sum().values.item()
                if nan_count > 0:
                    bad_splits["NaNs"].add(split)

        # report findings
        for reason, error_splits in bad_splits.items():
            if len(error_splits) > 0:
                print(f"Failed '{reason}': {dataset} for {error_splits}")


@app.command()
def random_subset(
    src_dataset: str,
    dest_dataset: str,
    pc: int = 50,
    split: str = "train",
    seed: int = 42,
):
    datasets_dir = Path(os.getenv("MOOSE_DERIVED_DATA")) / "nc-datasets"

    src_dataset_dir = datasets_dir / src_dataset
    dest_dataset_dir = datasets_dir / dest_dataset

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
    dataset_dir = Path(os.getenv("MOOSE_DERIVED_DATA")) / "nc-datasets" / dataset

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
    base_dir: Path = typer.Argument(..., envvar="MOOSE_DERIVED_DATA"),
):

    input_dir = os.path.join(base_dir, "nc-datasets", dataset)
    input_config_path = os.path.join(input_dir, "ds-config.yml")
    with open(input_config_path, "r") as f:
        config = yaml.safe_load(f)

    config_filters = config.get("filters", list())
    config_filters.append({"time_period": time_period})
    config["filters"] = config_filters

    output_dir = os.path.join(base_dir, "nc-datasets", f"{dataset}-{time_period}")
    os.makedirs(output_dir, exist_ok=False)
    for split_filepath in glob.glob(os.path.join(input_dir, "*.nc")):
        split_file = os.path.basename(split_filepath)
        logger.info(f"Filtering {split_file} to {output_dir}")
        split_ds = xr.open_dataset(split_filepath)
        output_filepath = os.path.join(output_dir, split_file)
        split_ds.sel(time=slice(*TIME_PERIODS[time_period])).to_netcdf(output_filepath)

    output_config_path = os.path.join(output_dir, "ds-config.yml")
    with open(output_config_path, "w") as f:
        yaml.dump(config, f)


@app.command()
def quantile(
    dataset: str,
    p: float,
    variable: str = "target_pr",
    base_dir: Path = typer.Argument(..., envvar="MOOSE_DERIVED_DATA"),
    split: str = "train",
):
    input_dir = os.path.join(base_dir, "nc-datasets", dataset)

    split_ds = xr.open_dataset(os.path.join(input_dir, f"{split}.nc"))
    Q_p = split_ds[variable].quantile(p)
    typer.echo(Q_p.values.item())
