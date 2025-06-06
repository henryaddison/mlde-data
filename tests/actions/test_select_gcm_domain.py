from importlib.resources import files
import pytest
import xarray as xr

from mlde_data.actions import get_action


def test_select_bham64_domain(global_dataset):
    bham_gcm_ds = get_action("select-subdomain")(domain="birmingham", size=9)(
        global_dataset
    )

    assert bham_gcm_ds["precipitation_flux"].size == 9 * 9
    assert bham_gcm_ds["longitude"].size == 9
    assert bham_gcm_ds["latitude"].size == 9


@pytest.fixture
def global_dataset():
    filepath = files("mlde_data.actions").joinpath(
        f"target_grids/60km/global/pr/moose_grid.nc"
    )
    return get_action("shift_lon_break")()(xr.open_dataset(filepath))
