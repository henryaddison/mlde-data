from importlib.resources import files
import os
from pathlib import Path
from typer.testing import CliRunner
import xarray as xr

from mlde_utils import VariableMetadata
from mlde_data.bin import app

runner = CliRunner()


def test_create(tmp_path):
    input_base_dir = Path(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "files",
        "variables",
        "raw",
        "moose",
    )
    output_base_dir = tmp_path

    collection = "land-cpm"
    frequency = "day"
    config_path = files("mlde_data").joinpath(
        f"../../config/variables/{frequency}/{collection}/predictors/temp.yml"
    )
    theta = "850"
    year = 1981
    ensemble_member = "01"
    domain = "engwales"
    scenario = "rcp85"
    scale_factor = "gcm"

    result = runner.invoke(
        app,
        [
            "variable",
            "create",
            "--config-path",
            str(config_path),
            "--theta",
            theta,
            "--scenario",
            scenario,
            "--ensemble-member",
            ensemble_member,
            "--year",
            str(year),
            "--domain",
            domain,
            "--scale-factor",
            scale_factor,
            "--input-base-dir",
            str(input_base_dir),
            "--output-base-dir",
            str(output_base_dir),
            "--no-validate",
        ],
    )
    assert result.exit_code == 0

    output_filepath = VariableMetadata(
        base_dir=output_base_dir,
        collection=collection,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable="temp850",
        frequency=frequency,
        resolution="2.2km-coarsened-gcm",
        domain=domain,
    ).filepath(year)

    xr.open_dataset(output_filepath)  # will raise error if file is invalid
