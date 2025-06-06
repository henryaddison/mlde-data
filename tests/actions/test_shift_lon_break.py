import numpy as np
import pytest
import xarray as xr

from mlde_data.actions import get_action


def test_shift_lon_break(global_dataset):
    orig_lon_attrs = global_dataset["longitude"].attrs

    ds = get_action("shift_lon_break")()(global_dataset)

    assert ds["longitude"].min().values.item() == -180.0
    assert ds["longitude"].max().values.item() == 170.0
    assert ds["longitude"].attrs == orig_lon_attrs


@pytest.fixture
def global_dataset():
    lon_attrs = {"axis": "X", "units": "degrees_east", "standard_name": "longitude"}
    longitude = xr.Variable(["longitude"], np.linspace(0, 350, 36), attrs=lon_attrs)

    latitude = xr.Variable(["latitude"], np.linspace(-90, 90, 19), attrs={})

    ds = xr.Dataset(
        {"foo": (("longitude", "latitude"), np.random.rand(36, 19))},
        coords={"longitude": longitude, "latitude": latitude},
    )

    return ds
