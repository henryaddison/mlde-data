from pathlib import Path
import pytest

from mlde_data.ceda_variable_adapter import CedaVariableAdapter


def test_hourly_filepaths(hourly_adapter):
    expected_dirpath = Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/01/pr/1hr")
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
    expected_dirpath = Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/01/pr/day")
    expected_filepaths = [
        Path(
            f"{expected_dirpath}/pr_rcp85_land-cpm_uk_2.2km_01_day_19811201-19811130.nc"
        )
    ]

    assert daily_adapter.filepaths == expected_filepaths


@pytest.fixture
def hourly_adapter():
    return CedaVariableAdapter(
        collection="land-cpm",
        ensemble_member="r001i1p00000",
        variable="pr",
        frequency="1hr",
        resolution="2.2km",
        domain="uk",
        scenario="rcp85",
        year=1981,
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
