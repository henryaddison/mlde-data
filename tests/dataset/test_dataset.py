import cftime
from mlde_utils import VariableMetadata
import numpy as np
import numpy.testing as npt
import os
import pytest
import xarray as xr

from mlde_data import dataset


@pytest.fixture
def config():
    return {
        "ensemble_members": ["r001i1p00000"],
        "scenario": "rcp85",
        "predictands": {
            "collection": "land-cpm",
            "domain": "test-10",
            "resolution": "2.2km",
            "frequency": "1hr",
            "variables": ["output1", "output2"],
        },
        "predictors": {
            "collection": "land-gcm",
            "domain": "test-10",
            "resolution": "60km",
            "frequency": "day",
            "variables": ["input1", "input2"],
        },
        "split": {
            "scheme": "random",
            "props": {"test": 0.2, "val": 0.2},
            "seed": 42,
            "time_periods": [["1980-12-01", "1981-11-30"]],
        },
    }


@pytest.fixture
def variable_files(tmp_path, config):
    derived_variables_path = tmp_path / "variables" / "derived"
    years = [1981]
    for em in config["ensemble_members"]:
        for var_type in ["predictands", "predictors"]:
            for var in config[var_type]["variables"]:
                if var_type == "predictands":
                    collection = "land-cpm"
                else:
                    collection = "land-gcm"
                meta = VariableMetadata(
                    derived_variables_path,
                    ensemble_member=em,
                    variable=var,
                    collection=collection,
                    scenario=config["scenario"],
                    domain=config[var_type]["domain"],
                    frequency=config[var_type]["frequency"],
                    resolution=config[var_type]["resolution"],
                )
                os.makedirs(meta.dirpath(), exist_ok=False)
                for year in years:
                    print(meta.filepath(year))
                    variable_ds_factory(var, year).to_netcdf(meta.filepath(year))

    return derived_variables_path


@pytest.fixture
def hourly_predictands_variable_files(tmp_path, config):
    derived_variables_path = tmp_path / "variables" / "derived"
    years = [1981]
    for em in config["ensemble_members"]:
        for var_type in ["predictands", "predictors"]:
            if var_type == "predictands":
                var_freq = "1hr"
            else:
                var_freq = "day"
            for var in config[var_type]["variables"]:
                if var_type == "predictands":
                    collection = "land-cpm"
                else:
                    collection = "land-gcm"
                meta = VariableMetadata(
                    derived_variables_path,
                    ensemble_member=em,
                    variable=var,
                    collection=collection,
                    scenario=config["scenario"],
                    domain=config[var_type]["domain"],
                    frequency=config[var_type]["frequency"],
                    resolution=config[var_type]["resolution"],
                )
                os.makedirs(meta.dirpath(), exist_ok=False)
                for year in years:
                    print(meta.filepath(year))
                    variable_ds_factory(var, year, freq=var_freq).to_netcdf(
                        meta.filepath(year)
                    )

    return derived_variables_path


def variable_ds_factory(var, year, freq="day"):
    if freq == "day":
        time_range = xr.date_range(
            cftime.Datetime360Day(year - 1, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=360,
            freq="D",
            use_cftime=True,
        )
    elif freq == "1hr":
        time_range = xr.date_range(
            cftime.Datetime360Day(year - 1, 12, 1, 0, 0, 0, 0, has_year_zero=True),
            periods=360 * 24,
            freq="H",
            use_cftime=True,
        )
    else:
        raise ValueError(f"Unsupported frequency {freq}")
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
    ds["time"].attrs["axis"] = "T"
    ds["grid_latitude"].attrs["axis"] = "Y"
    ds["grid_longitude"].attrs["axis"] = "X"

    return ds


def test_single_variable(variable_files, config):
    input_base_dir = variable_files
    var_name = "input1"
    var_config = {k: config[k] for k in ["scenario"]} | {
        k: config["predictors"][k]
        for k in ["resolution", "collection", "frequency", "domain"]
    }

    result = dataset._single_variable(
        "r001i1p00000", var_name, input_base_dir, **var_config
    )

    assert result.sizes == {
        "ensemble_member": 1,
        "time": 360,
        "grid_longitude": 10,
        "grid_latitude": 10,
    }
    assert result["input1"].shape == (1, 360, 10, 10)


def test_create(variable_files, config):
    input_base_dir = variable_files

    result, _ = dataset.create(config, input_base_dir)

    assert set(result.keys()) == {"predictors", "predictands"}

    for var_type in ["predictors", "predictands"]:

        ds = result[var_type]["train"]

        assert ds.sizes == {
            "ensemble_member": 1,
            "time": 6 * 360 / 10,
            "grid_longitude": 10,
            "grid_latitude": 10,
        }
        for var_name in config[var_type]["variables"]:
            assert ds[var_name].shape == (1, 216, 10, 10)


def test_create_statistics(variable_files, config):
    input_base_dir = variable_files

    splits, stats = dataset.create(config, input_base_dir)

    assert set(stats.keys()) == {"predictors", "predictands"}

    for var_type in ["predictors", "predictands"]:
        split = "train"
        for var_name in config[var_type]["variables"]:
            expected = np.mean(splits[var_type][split][var_name].values)
            actual = (
                stats[var_type]["train"]
                .sel(variable=var_name, time_aggregation_factor=1)["mean"]
                .values
            )
            npt.assert_equal(actual, expected)

            expected = np.std(splits[var_type][split][var_name].values)
            actual = (
                stats[var_type]["train"]
                .sel(variable=var_name, time_aggregation_factor=1)["std"]
                .values
            )
            npt.assert_equal(actual, expected)

            expected = np.size(splits[var_type][split][var_name].values)
            actual = (
                stats[var_type]["train"]
                .sel(variable=var_name, time_aggregation_factor=1)["count"]
                .values
            )
            npt.assert_equal(actual, expected)

            expected = np.max(splits[var_type][split][var_name].values)
            actual = (
                stats[var_type]["train"]
                .sel(variable=var_name, time_aggregation_factor=1)["max"]
                .values
            )
            npt.assert_equal(actual, expected)

            expected = np.min(splits[var_type][split][var_name].values)
            actual = (
                stats[var_type]["train"]
                .sel(variable=var_name, time_aggregation_factor=1)["min"]
                .values
            )
            npt.assert_equal(actual, expected)


def test_create_time_aggregated_statistics(hourly_predictands_variable_files, config):
    input_base_dir = hourly_predictands_variable_files
    time_factors = [1, 3, 6, 24]
    config["predictands"]["stats"] = {"time_aggregation_factors": time_factors}

    splits, stats = dataset.create(config, input_base_dir)

    assert set(stats.keys()) == {"predictors", "predictands"}

    for var_type in ["predictors", "predictands"]:
        split = "train"
        for var_name in config[var_type]["variables"]:
            for time_factor in config[var_type].get(
                "stats", {"time_aggregation_factors": [1]}
            )["time_aggregation_factors"]:
                for metric_name, metric_exp_func in {
                    "mean": np.mean,
                    "std": np.std,
                    "count": np.size,
                    "max": np.max,
                    "min": np.min,
                }.items():
                    if time_factor == 1:
                        expected = metric_exp_func(
                            splits[var_type][split][var_name].values
                        )
                    else:
                        expected = metric_exp_func(
                            splits[var_type][split][var_name]
                            .coarsen(time=time_factor)
                            .sum()
                            .values
                        )
                    actual = (
                        stats[var_type]["train"]
                        .sel(variable=var_name, time_aggregation_factor=time_factor)[
                            metric_name
                        ]
                        .values
                    )
                    npt.assert_equal(actual, expected)
