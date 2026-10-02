from importlib.resources import files
from mlde_utils import VariableMetadata
import os
from pathlib import Path
import pytest

from mlde_data.bin.variable import create
from mlde_data.options import CollectionOption, DomainOption
from mlde_data.variable.validation import validate


def test_validate_gcmx(variable_gcmx):
    validation_errors = validate(variable_gcmx[0], year=variable_gcmx[1])
    print(validation_errors)
    assert validation_errors == ["time shape"]


def test_validate_22cpm(variable_22cpm):
    validation_errors = validate(variable_22cpm[0], year=variable_22cpm[1])
    print(validation_errors)
    assert validation_errors == ["time shape"]


def test_validate_5cpm(variable_5cpm):
    validation_errors = validate(variable_5cpm[0], year=variable_5cpm[1])
    print(validation_errors)
    assert validation_errors == ["time shape"]


@pytest.fixture
def variable_gcmx(tmp_path):
    domain = DomainOption.uk
    collection = CollectionOption.cpm
    frequency = "day"
    year = 1982
    ensemble_member = "r001i1p00000"
    theta = "850"
    scale_factor = "gcm"
    var_type = "temp"
    scenario = "rcp85"

    config_path = files("mlde_data").joinpath(
        f"../../config/variables/{frequency}/{collection.value}/predictors/{var_type}.yml"
    )

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

    create(
        config_paths=[config_path],
        thetas=[theta],
        scenario=scenario,
        ensemble_member=ensemble_member,
        year=year,
        scale_factor=scale_factor,
        domain=domain,
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir,
        validate=False,
    )

    variable = f"{var_type}{theta}"

    return (
        VariableMetadata(
            base_dir=output_base_dir,
            collection=collection.value,
            scenario=scenario,
            ensemble_member=ensemble_member,
            variable=variable,
            frequency=frequency,
            resolution="2.2km-coarsened-gcm",
            domain=domain.value,
        ),
        year,
    )


@pytest.fixture
def variable_22cpm(tmp_path):
    domain = DomainOption.engwales
    collection = CollectionOption.cpm
    frequency = "1hr"
    year = 1981
    ensemble_member = "r001i1p00000"
    scale_factor = 4
    var_type = "pr"
    scenario = "rcp85"

    config_path = files("mlde_data").joinpath(
        f"../../config/variables/{frequency}/{collection.value}/targets/{var_type}.yml"
    )

    input_base_dir = Path(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "files",
        "variables",
        "raw",
        "ceda",
        "badc",
        "ukcp18",
        "data",
    )

    output_base_dir = tmp_path

    create(
        config_paths=[config_path],
        scenario=scenario,
        ensemble_member=ensemble_member,
        year=year,
        scale_factor=scale_factor,
        domain=domain,
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir,
        validate=False,
    )

    variable = var_type

    return (
        VariableMetadata(
            base_dir=output_base_dir,
            collection=collection.value,
            scenario=scenario,
            ensemble_member=ensemble_member,
            variable=variable,
            frequency=frequency,
            resolution="2.2km-coarsened-4x",
            domain=domain.value,
        ),
        year,
    )


@pytest.fixture
def variable_5cpm(tmp_path):
    domain = DomainOption.uk
    collection = CollectionOption.cpm
    frequency = "1hr"
    year = 1981
    ensemble_member = "r001i1p00000"
    scale_factor = 1
    var_type = "pr"
    scenario = "rcp85"

    config_path = files("mlde_data").joinpath(
        f"../../config/variables/{frequency}/{collection.value}/targets/5km/{var_type}.yml"
    )

    input_base_dir = Path(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "files",
        "variables",
        "raw",
        "ceda",
        "badc",
        "ukcp18",
        "data",
    )

    output_base_dir = tmp_path

    create(
        config_paths=[config_path],
        scenario=scenario,
        ensemble_member=ensemble_member,
        year=year,
        scale_factor=scale_factor,
        domain=domain,
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir,
        validate=False,
    )

    variable = var_type

    return (
        VariableMetadata(
            base_dir=output_base_dir,
            collection=collection.value,
            scenario=scenario,
            ensemble_member=ensemble_member,
            variable=variable,
            frequency=frequency,
            resolution="5km",
            domain=domain.value,
        ),
        year,
    )
