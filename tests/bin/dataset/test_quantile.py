import os

import numpy as np
import pytest
from typer.testing import CliRunner
import xarray as xr
import yaml

from mlde_data.bin import app

runner = CliRunner()


def test_quantile(tmp_path, dataset, time_range, lat_range, lon_range):
    result = runner.invoke(
        app,
        [
            "dataset",
            "quantile",
            dataset,
            str(0.5),
            str(tmp_path),
            "--variable",
            "measurement",
        ],
    )
    assert result.exit_code == 0
    assert (
        f"{(len(time_range) * len(lat_range) * len(lon_range) - 1) * 0.5}\n"
        == result.stdout
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "quantile",
            dataset,
            str(0.25),
            str(tmp_path),
            "--variable",
            "measurement",
        ],
    )
    assert result.exit_code == 0
    assert (
        f"{(len(time_range) * len(lat_range) * len(lon_range) - 1) * 0.25}\n"
        == result.stdout
    )


@pytest.fixture
def time_range():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def lat_range():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def lon_range():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def dataset(tmp_path, time_range, lat_range, lon_range):
    dataset_name = "test-dataset"

    train_ds = xr.Dataset(
        data_vars={
            "measurement": (
                ["time", "grid_longitude", "grid_latitude"],
                np.arange(len(time_range) * len(lat_range) * len(lon_range)).reshape(
                    len(time_range), len(lat_range), len(lon_range)
                ),
            ),
        },
        coords=dict(
            time=(["time"], time_range),
            grid_longitude=(["grid_longitude"], lon_range),
            grid_latitude=(["grid_latitude"], lat_range),
        ),
    )

    dataset_path = tmp_path / dataset_name
    os.makedirs(dataset_path, exist_ok=True)
    train_ds.to_netcdf(dataset_path / "train.nc")

    config = {}
    config_path = os.path.join(dataset_path, "ds-config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return dataset_name
