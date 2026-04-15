import cftime
import os
from pathlib import Path
import pytest

from mlde_data.ceda_variable_adapter import CedaVariableAdapter
from mlde_data.variable import SourceVariableConfig


def test_from_variable_defn(hourly_defn):
    adapter = CedaVariableAdapter.from_variable_defn(
        hourly_defn, ensemble_member="r001i1p00000", scenario="rcp85", year=1981
    )

    assert adapter.collection == "land-cpm"
    assert adapter.variable == "pr"
    assert adapter.frequency == "1hr"
    assert adapter.resolution == "2.2km"
    assert adapter.domain == "uk"
    assert adapter.ensemble_member == "r001i1p00000"
    assert adapter.scenario == "rcp85"
    assert adapter.year == 1981


def test_eq(hourly_adapter, hourly_defn):
    adapter = CedaVariableAdapter.from_variable_defn(
        hourly_defn, ensemble_member="r001i1p00000", scenario="rcp85", year=1981
    )
    assert adapter == hourly_adapter


def test_hourly_filepaths(hourly_adapter, fixtures_base_dir):
    expected_dirpath = Path(
        f"{fixtures_base_dir}/land-cpm/uk/2.2km/rcp85/01/pr/1hr/v20210615"
    )
    expected_filepaths = [
        Path(
            f"{expected_dirpath}/pr_rcp85_land-cpm_uk_2.2km_01_1hr_19801201-19801230.nc"
        )
    ] + [
        Path(
            f"{expected_dirpath}/pr_rcp85_land-cpm_uk_2.2km_01_1hr_1981{month:02d}01-1981{month:02d}30.nc"
        )
        for month in range(1, 12)
    ]

    assert hourly_adapter.filepaths == expected_filepaths


def test_daily_filepaths(daily_adapter):
    expected_dirpath = Path(
        "/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/01/pr/day/v20210615"
    )
    expected_filepaths = [
        Path(
            f"{expected_dirpath}/pr_rcp85_land-cpm_uk_2.2km_01_day_19811201-19811130.nc"
        )
    ]

    assert daily_adapter.filepaths == expected_filepaths


def test_open(hourly_adapter):
    ds = hourly_adapter.open()
    assert "pr" in ds.data_vars
    assert ds["time"].min().item() == cftime.Datetime360Day(
        1980, 12, 1, 0, 30, 0, 0, has_year_zero=True
    )
    assert ds["time"].max().item() == cftime.Datetime360Day(
        1981, 11, 30, 4, 30, 0, 0, has_year_zero=True
    )
    assert "ensemble_member" not in ds.dims


@pytest.fixture
def fixtures_base_dir():
    return Path(
        os.path.dirname(__file__),
        "fixtures",
        "files",
        "variables",
        "raw",
        "ceda",
        "badc",
        "ukcp18",
        "data",
    )


@pytest.fixture
def hourly_adapter(fixtures_base_dir):
    return CedaVariableAdapter(
        collection="land-cpm",
        ensemble_member="r001i1p00000",
        variable="pr",
        frequency="1hr",
        resolution="2.2km",
        domain="uk",
        scenario="rcp85",
        year=1981,
        base_dir=fixtures_base_dir,
    )


@pytest.fixture
def daily_adapter():
    return CedaVariableAdapter(
        collection="land-cpm",
        ensemble_member="r001i1p00000",
        variable="pr",
        frequency="day",
        resolution="2.2km",
        domain="uk",
        scenario="rcp85",
        year=1981,
    )


@pytest.fixture
def hourly_defn():
    return SourceVariableConfig(
        src_type="ceda",
        collection="land-cpm",
        frequency="1hr",
        variable="pr",
    )
