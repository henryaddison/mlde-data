import os

import cftime
import numpy as np
import pytest
from typer.testing import CliRunner
import xarray as xr
import yaml

from mlde_data.bin import app

runner = CliRunner()


def test_filter(tmp_path, dataset):
    period = "historic"
    # assert dataset_path == ""
    dataset
    result = runner.invoke(app, ["dataset", "filter", dataset, period, str(tmp_path)])
    assert result.exit_code == 0


@pytest.fixture
def dataset(tmp_path):
    dataset_name = "test-dataset"
    test_time_range = (
        xr.date_range(
            cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=360 * 20,
            freq="D",
            use_cftime=True,
        )
        .append(
            xr.date_range(
                cftime.Datetime360Day(2020, 12, 1, 12, 0, 0, 0, has_year_zero=True),
                periods=360 * 20,
                freq="D",
                use_cftime=True,
            )
        )
        .append(
            xr.date_range(
                cftime.Datetime360Day(2060, 12, 1, 12, 0, 0, 0, has_year_zero=True),
                periods=360 * 20,
                freq="D",
                use_cftime=True,
            )
        )
    )

    train_time_range = (
        xr.date_range(
            cftime.Datetime360Day(1999, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=360 * 20,
            freq="D",
            use_cftime=True,
        )
        .append(
            xr.date_range(
                cftime.Datetime360Day(2039, 12, 1, 12, 0, 0, 0, has_year_zero=True),
                periods=360 * 20,
                freq="D",
                use_cftime=True,
            )
        )
        .append(
            xr.date_range(
                cftime.Datetime360Day(2079, 12, 1, 12, 0, 0, 0, has_year_zero=True),
                periods=360 * 20,
                freq="D",
                use_cftime=True,
            )
        )
    )

    lat_range = np.linspace(-2, 2, 10)
    lon_range = np.linspace(-2, 2, 10)

    test_ds = xr.Dataset(
        data_vars={
            "measurement": (
                ["time", "grid_longitude", "grid_latitude"],
                np.ones([len(test_time_range), len(lat_range), len(lon_range)]),
            ),
        },
        coords=dict(
            time=(["time"], test_time_range),
            grid_longitude=(["grid_longitude"], lon_range),
            grid_latitude=(["grid_latitude"], lat_range),
        ),
    )

    train_ds = xr.Dataset(
        data_vars={
            "measurement": (
                ["time", "grid_longitude", "grid_latitude"],
                np.ones([len(train_time_range), len(lat_range), len(lon_range)]),
            ),
        },
        coords=dict(
            time=(["time"], train_time_range),
            grid_longitude=(["grid_longitude"], lon_range),
            grid_latitude=(["grid_latitude"], lat_range),
        ),
    )

    dataset_path = tmp_path / dataset_name
    os.makedirs(dataset_path, exist_ok=True)
    test_ds.to_netcdf(dataset_path / "test.nc")
    train_ds.to_netcdf(dataset_path / "train.nc")

    config = {}
    config_path = os.path.join(dataset_path, "ds-config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return dataset_name
