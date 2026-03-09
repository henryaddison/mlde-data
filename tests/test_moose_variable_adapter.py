import os
from pathlib import Path
import pytest

from mlde_data.moose_variable_adapter import MooseVariableAdapter


def test_open(fixtures_base_dir):
    adapter = MooseVariableAdapter(
        collection="land-cpm",
        ensemble_member="r001i1p00000",
        variable="psl",
        frequency="day",
        resolution="2.2km",
        domain="uk",
        scenario="rcp85",
        year=1982,
        base_dir=fixtures_base_dir,
    )
    ds = adapter.open()
    assert ds["time"].shape == (2,)


@pytest.fixture
def fixtures_base_dir():
    return Path(
        os.path.dirname(__file__),
        "fixtures",
        "files",
        "variables",
        "raw",
        "moose",
    )
