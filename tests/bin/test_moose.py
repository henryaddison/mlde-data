import iris
import numpy as np
import os
from pathlib import Path
from typer.testing import CliRunner

from mlde_data.bin import app
from mlde_data.bin.moose import MoosePPVariableMetadata
from mlde_utils import VariableMetadata

runner = CliRunner()


def test_convert(tmp_path):
    input_base_dir = Path(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "files",
        "variables",
        "raw",
        "moose",
        "pp",
    )
    output_base_dir = tmp_path

    result = runner.invoke(
        app,
        [
            "moose",
            "convert",
            "--collection",
            "land-cpm",
            "--scenario",
            "rcp85",
            "--ensemble-member",
            "r001i1p00000",
            "--year",
            "1981",
            "--variable",
            "mlqtw",
            "--frequency",
            "day",
            "--input-base-dir",
            str(input_base_dir),
            "--output-base-dir",
            str(output_base_dir),
            "--no-validate",
        ],
    )
    assert result.exit_code == 0

    input_glob = MoosePPVariableMetadata(
        base_dir=input_base_dir,
        collection="land-cpm",
        scenario="rcp85",
        ensemble_member="r001i1p00000",
        variable="mlqtw",
        frequency="day",
        resolution="2.2km",
        domain="uk",
    ).pp_files_glob(1981)

    output_filepath = VariableMetadata(
        base_dir=output_base_dir,
        collection="land-cpm",
        scenario="rcp85",
        ensemble_member="r001i1p00000",
        variable="mlqtw",
        frequency="day",
        resolution="2.2km",
        domain="uk",
    ).filepath(1981)

    for c1, c2 in zip(iris.load(str(output_filepath)), iris.load(input_glob)):
        assert np.all(c1.data == c2.data)
