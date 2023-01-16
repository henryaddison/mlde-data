from collections import defaultdict
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

from ml_downscaling_emulator import VariableMetadata
from ml_downscaling_emulator.data.dataset import (
    RandomSplit,
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
    val_prop: float = 0.2,
    test_prop: float = 0.1,
):
    """
    Create a dataset
    """
    config_name = config.stem
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    predictand_var_params = {
        k: config[k] for k in ["domain", "ensemble_member", "scenario", "frequency"]
    }
    predictand_var_params.update(
        {
            "variable": config["predictand"]["variable"],
            "resolution": config["predictand"]["resolution"],
        }
    )
    predictand_meta = VariableMetadata(input_base_dir, **predictand_var_params)

    predictors_meta = []
    for predictor_var_config in config["predictors"]:
        var_params = {
            k: config[k]
            for k in [
                "domain",
                "ensemble_member",
                "scenario",
                "frequency",
                "resolution",
            ]
        }
        var_params.update({k: predictor_var_config[k] for k in ["variable"]})
        predictors_meta.append(VariableMetadata(input_base_dir, **var_params))

    example_predictor_filepath = predictors_meta[0].existing_filepaths()[0]
    time_encoding = xr.open_dataset(example_predictor_filepath).time_bnds.encoding

    predictor_datasets = [
        xr.open_mfdataset(dsmeta.existing_filepaths()) for dsmeta in predictors_meta
    ]
    predictand_dataset = xr.open_mfdataset(predictand_meta.existing_filepaths()).rename(
        {predictand_meta.variable: f"target_{predictand_meta.variable}"}
    )

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

    if config["split_scheme"] == "ssi":
        splitter = SeasonStratifiedIntensitySplit(
            val_prop=val_prop, test_prop=test_prop, time_encoding=time_encoding
        )
    elif config["split_scheme"] == "random":
        splitter = RandomSplit(
            val_prop=val_prop, test_prop=test_prop, time_encoding=time_encoding
        )
    else:
        raise RuntimeError(f"Unknown split scheme {config['split_scheme']}")

    split_sets = splitter.run(combined_dataset)

    output_dir = os.path.join(output_base_dir, "nc-datasets", config_name)

    os.makedirs(output_dir, exist_ok=False)

    logger.info(f"Saving data to {output_dir}")
    with open(os.path.join(output_dir, "ds-config.yml"), "w") as f:
        yaml.dump(config, f)
    for split_name, split_ds in split_sets.items():
        split_ds.to_netcdf(os.path.join(output_dir, f"{split_name}.nc"))


@app.command()
def validate():
    datasets = [
        "2.2km-coarsened-8x_london_vorticity850_random",
        "2.2km-coarsened-gcm-2.2km-coarsened-4x_birmingham_vorticity850_random",
        "bham_gcmx-4x_psl-spechum-temp-vort_random",
        "bham_gcmx-4x_psl-temp-vort_random",
        "bham_gcmx-4x_psl-tempgrad-vort_random",
        "bham_gcmx-4x_spechum-temp-vorticity850_random",
        "bham_gcmx-4x_spechum-temp-vort_random",
        "bham_gcmx-4x_temp-vort_random",
        "bham_gcmx-4x_tempgrad-vort_random",
        "bham_gcmx-4x_vort850_random",
        "bham_gcmx-4x_pr_random",
        "bham_gcmx-4x_linpr_random",
        "60km-2.2km-coarsened-4x_birmingham_vorticity850_random",
        "bham_60km-4x_psl-spechum-temp-vort_random",
        "bham_60km-4x_psl-temp-vort_random",
        "bham_60km-4x_psl-tempgrad-vort_random",
        "bham_60km-4x_spechum-temp-vorticity850_random",
        "bham_60km-4x_spechum-temp-vort_random",
        "bham_60km-4x_temp-vort_random",
        "bham_60km-4x_tempgrad-vort_random",
        "bham_60km-4x_vort850_random",
        "bham_60km-4x_pr_random",
        "bham_60km-4x_linpr_random",
        "60km-2.2km_london_vorticity850_random",
    ]

    splits = ["train", "val", "test"]

    for dataset in datasets:

        bad_splits = defaultdict(set)
        for split in splits:
            sys.stdout.write("\033[K")
            print(f"Checking {split} of {dataset}", end="\r")
            dataset_path = os.path.join(
                os.getenv("MOOSE_DERIVED_DATA"), "nc-datasets", dataset, f"{split}.nc"
            )
            try:
                ds = xr.open_dataset(dataset_path)
            except FileNotFoundError:
                bad_splits["no file"].add(split)
                continue

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
