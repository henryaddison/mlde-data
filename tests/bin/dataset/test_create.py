from mlde_utils import DatasetMetadata
import os
from pathlib import Path
import pytest
from typer.testing import CliRunner
import xarray as xr

from mlde_data.bin import app
from mlde_data.bin.dataset import create

runner = CliRunner()


@pytest.fixture
def config_filepath():
    return Path(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fixtures",
            "files",
            "config",
            "dataset",
            "test_dataset.yml",
        )
    )


@pytest.fixture
def input_base_dir():
    return Path(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fixtures",
            "files",
            "variables",
            "derived",
        )
    )


def assert_file(path, exists=True):
    if exists:
        assert os.path.exists(path), f"Expected file {path} does not exist"
    else:
        assert not os.path.exists(path), f"File {path} should not exist but does"


def test_create(tmp_path, config_filepath, input_base_dir):
    expected_dsmeta = DatasetMetadata(config_filepath.stem, base_dir=tmp_path)

    ds_config_filepath = expected_dsmeta.config_path()

    assert_file(ds_config_filepath, exists=False)
    create(
        config_filepath,
        input_base_dir=input_base_dir,
        output_base_dir=tmp_path,
    )
    assert_file(ds_config_filepath)

    for var_type in ["predictors", "predictands"]:
        for split in ["train", "val", "test"]:
            data_filepath = expected_dsmeta.path() / split / f"{var_type}.zarr"
            assert_file(data_filepath)
            xr.open_dataset(data_filepath)  # will raise error if file is invalid


def test_create_runner(tmp_path, config_filepath, input_base_dir):
    result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            str(config_filepath),
            str(input_base_dir),
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
