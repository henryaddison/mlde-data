import cftime
from mlde_utils import VariableMetadata
import numpy as np
import os
import pytest
import xarray as xr

from mlde_data import dataset


@pytest.fixture
def config():
    return {
        "domain": "test-10",
        "frequency": "day",
        "ensemble_members": ["01"],
        "scenario": "rcp85",
        "predictands": {
            "resolution": "2.2km",
            "variables": ["output1", "output2"],
        },
        "predictors": {
            "resolution": "60km",
            "variables": ["input1", "input2"],
        },
        "split": {
            "scheme": "random-season",
            "test_prop": 0.2,
            "val_prop": 0.2,
            "seed": 42,
        },
    }


@pytest.fixture
def variable_files(tmp_path, config):
    years = [1981]
    for em in config["ensemble_members"]:
        for var_type in ["predictands", "predictors"]:
            for var in config[var_type]["variables"]:
                print(var)
                meta = VariableMetadata(
                    tmp_path / "moose",
                    ensemble_member=em,
                    domain=config["domain"],
                    variable=var,
                    resolution=config[var_type]["resolution"],
                    scenario=config["scenario"],
                    frequency=config["frequency"],
                )
                os.makedirs(meta.dirpath(), exist_ok=False)
                for year in years:
                    print(meta.filepath(year))
                    variable_ds_factory(var, year).to_netcdf(meta.filepath(year))

    return tmp_path


def variable_ds_factory(var, year):
    time_range = xr.cftime_range(
        cftime.Datetime360Day(year - 1, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360,
        freq="D",
    )
    grid_lat_range = np.linspace(-2, 2, 10)
    grid_lon_range = np.linspace(-2, 2, 10)

    ds = xr.Dataset(
        data_vars={
            var: (
                ["time", "grid_longitude", "grid_latitude"],
                np.random.randn(
                    len(time_range), len(grid_lat_range), len(grid_lon_range)
                ),
            ),
        },
        coords=dict(
            time=(["time"], time_range),
            grid_longitude=(["grid_longitude"], grid_lon_range),
            grid_latitude=(["grid_latitude"], grid_lat_range),
        ),
    )

    return ds


def test_combine_variables(variable_files, config):
    input_base_dir = variable_files

    result = dataset._combine_variables("01", config, input_base_dir)

    assert result.dims == {
        "ensemble_member": 1,
        "time": 360,
        "grid_longitude": 10,
        "grid_latitude": 10,
    }
    assert result["input1"].shape == (1, 360, 10, 10)
    assert result["input2"].shape == (1, 360, 10, 10)
    assert result["target_output1"].shape == (1, 360, 10, 10)
    assert result["target_output2"].shape == (1, 360, 10, 10)
