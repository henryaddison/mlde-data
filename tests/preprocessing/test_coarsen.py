import os

import pytest
import xarray as xr

from ml_downscaling_emulator.preprocessing.coarsen import Coarsen


@pytest.mark.skip(reason="not getting the expected coarsen calculation correct")
def test_coarsen():
    ds_filepath = os.path.join(
        os.path.dirname(__file__), "..", "example_moose_extract.nc"
    )
    ds = xr.load_dataset(ds_filepath)

    coarsened = Coarsen(scale_factor=2, variable="air_pressure_at_sea_level").run(ds)

    expected_value = ds.air_pressure_at_sea_level.values[0:2, 0:2, 0].mean()

    coarsened_value = coarsened.isel(
        grid_longitude=0, grid_latitude=0, time=0
    ).air_pressure_at_sea_level.values

    assert coarsened_value == expected_value
