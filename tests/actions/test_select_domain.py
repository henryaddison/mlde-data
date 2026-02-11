from importlib.resources import files
import os
from pathlib import Path
import pytest
import xarray as xr

from mlde_data.actions import get_action
from mlde_data.moose import open_pp_data
from mlde_data.options import CollectionOption


def test_select_bham64_domain(global_dataset):
    engwales_ds = get_action("select-subdomain")(domain="engwales")(global_dataset)

    assert engwales_ds["precipitation_flux"].size == 13 * 13
    assert engwales_ds["longitude"].size == 13
    assert engwales_ds["latitude"].size == 13


def test_select_engwales_domain(moose_cpm_dataset):
    engwales_ds = get_action("select-subdomain")(domain="engwales")(moose_cpm_dataset)

    assert engwales_ds["temp850"].size == 256 * 256 * engwales_ds.cf["T"].size
    assert engwales_ds.cf["X"].size == 256
    assert engwales_ds.cf["Y"].size == 256


@pytest.fixture
def global_dataset():
    filepath = files("mlde_data.actions").joinpath(
        f"target_grids/60km/global/pr/moose_grid.nc"
    )
    return get_action("shift_lon_break")()(
        xr.open_dataset(filepath).assign_attrs(
            {
                "domain": "global",
                "resolution": "60km",
                "frequency": "day",
            }
        )
    )


@pytest.fixture
def moose_cpm_dataset():
    base_dir = Path(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "fixtures",
            "files",
            "variables",
            "raw",
            "moose",
            "pp",
        )
    )
    return (
        open_pp_data(
            base_dir=base_dir,
            collection=CollectionOption.cpm,
            scenario="rcp85",
            ensemble_member="r001i1p00000",
            variable="mlqtw",
            frequency="day",
            resolution="2.2km",
            domain="uk",
            year=1981,
        )
        .assign_attrs(
            {
                "domain": "uk",
                "resolution": "2.2km",
                "frequency": "day",
            }
        )
        .sel(pressure=850)
        .rename({"air_temperature": "temp850"})
    )
