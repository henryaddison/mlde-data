import glob
from importlib.resources import files
import logging
import numpy as np
import os
from pathlib import Path
import shutil
import sys
import typer
import xarray as xr
import yaml


from mlde_utils import (
    TIME_PERIODS,
    dataset_path,
    dataset_config_path,
    dataset_config,
)

from .. import dataset as dataset_lib

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
    Create and save a dataset
    """
    config_name = config.stem
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    split_sets = dataset_lib.create(config, input_base_dir)

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


def report_issues(dataset, bad_splits):
    for reason, error_splits in bad_splits.items():
        if len(error_splits) > 0:
            print(f"Failed '{reason}': {dataset} for {error_splits}")


@app.command()
def validate(dataset_name: str = typer.Argument("all")):
    datasets = list(
        map(
            lambda f: f.stem,
            files("mlde_data.config").joinpath("datasets").glob("*.yml"),
        )
    )

    if dataset_name != "all":
        if dataset_name not in datasets:
            logger.warning(
                f"Dataset {dataset_name} not found in standard list. Continuing but may not be valid."
            )
        datasets = [dataset_name]

    for dataset in datasets:
        sys.stdout.write("\033[K")
        print(f"Checking {dataset}", end="\r")
        bad_splits = dataset_lib.validate(dataset)
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
