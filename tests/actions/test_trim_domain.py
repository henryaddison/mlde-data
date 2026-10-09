import numpy as np
import pytest
import xarray as xr

from mlde_data.actions import get_action


def test_trim_domain(dataset):
    ds = get_action("trim-domain")(patch_size=16)(dataset)

    # trim-domain should turn dataset into a multiple of patch size (16)
    assert ds["longitude"].size == 32
    assert ds["latitude"].size == 16

    # by removing first entries in X and Y coords
    assert ds["longitude"][0].item() == dataset["longitude"][4].item()
    assert ds["latitude"][0].item() == dataset["latitude"][3].item()


@pytest.fixture
def dataset():
    lon_attrs = {"axis": "X", "units": "degrees_east", "standard_name": "longitude"}
    longitude = xr.Variable(["longitude"], np.linspace(0, 350, 36), attrs=lon_attrs)

    lat_attrs = {"axis": "Y", "units": "degrees_north", "standard_name": "latitude"}
    latitude = xr.Variable(["latitude"], np.linspace(-90, 90, 19), attrs=lat_attrs)

    ds = xr.Dataset(
        {"foo": (("longitude", "latitude"), np.random.rand(36, 19))},
        coords={"longitude": longitude, "latitude": latitude},
    )

    return ds
