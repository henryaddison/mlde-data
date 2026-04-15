import logging
import os
from pathlib import Path
from mlde_utils import RAW_MOOSE_VARIABLES_PATH
from mlde_data.options import DomainOption, CollectionOption
from mlde_data.bin.moose import extract, clean
from mlde_data.bin.variable import create as create_variable
from mlde_data.variable import load_config
from mlde_utils import VariableMetadata
import typer
from typing import List


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def moose(
    years: List[int],
    variable_configs: List[Path] = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    thetas: List[int] = None,
    target_resolution: str = None,
    force: bool = False,
    cleanup: bool = True,
):

    configs = [
        load_config(
            variable_config,
            scale_factor=scale_factor,
            domain=domain.value,
            theta=theta,
            target_resolution=target_resolution,
        )
        for variable_config in variable_configs
        for theta in (thetas or [None])
    ]

    src_configs = {src_config for config in configs for src_config in config["sources"]}

    src_type = {src_config.src_type for src_config in src_configs}
    # TODO: support creating variables from multiple source types
    assert (
        len(src_type) == 1
    ), "All source variable configs must have the same source type"
    src_type = src_type.pop()
    assert (
        src_type == "moose"
    ), "Only moose source variables supported for moose command"

    for year in years:
        # only moose sources need to extract data first (for others assumed on accessible filesystem)
        for src_config in src_configs:
            source_nc_filepath = VariableMetadata(
                base_dir=RAW_MOOSE_VARIABLES_PATH,
                variable=src_config.variable,
                frequency=src_config.frequency,
                domain=src_config.domain,
                resolution=src_config.resolution,
                ensemble_member=ensemble_member,
                scenario=scenario,
                collection=src_config.collection,
            ).filepath(year)
            # skip extract if file already exists and not forcing an extraction
            if os.path.exists(source_nc_filepath) and not force:
                logger.info(f"{source_nc_filepath} already exists, skipping extraction")
                continue

            extract(
                variable=src_config.variable,
                year=year,
                frequency=src_config.frequency,
                collection=CollectionOption(src_config.collection),
                ensemble_member=ensemble_member,
                scenario=scenario,
            )

        # run create variable
        create_variable(
            config_paths=variable_configs,
            year=year,
            domain=domain,
            scale_factor=scale_factor,
            ensemble_member=ensemble_member,
            scenario=scenario,
            thetas=thetas,
            target_resolution=target_resolution,
        )

        # run clean up for moose extracts
        if cleanup:
            for src_config in src_configs:
                clean(
                    collection=CollectionOption(src_config.collection),
                    scenario=scenario,
                    ensemble_member=ensemble_member,
                    variable=src_config.variable,
                    frequency=src_config.frequency,
                    year=year,
                )


@app.command()
def ceda(
    years: List[int],
    variable_configs: List[Path] = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    thetas: List[int] = None,
    target_resolution: str = None,
    force: bool = False,
    cleanup: bool = True,
):

    configs = [
        load_config(
            variable_config,
            scale_factor=scale_factor,
            domain=domain.value,
            theta=theta,
            target_resolution=target_resolution,
        )
        for variable_config in variable_configs
        for theta in (thetas or [None])
    ]

    src_configs = {src_config for config in configs for src_config in config["sources"]}

    src_type = {src_config.src_type for src_config in src_configs}
    # TODO: support creating variables from multiple source types
    assert (
        len(src_type) == 1
    ), "All source variable configs must have the same source type"
    src_type = src_type.pop()
    assert src_type == "ceda", "Only ceda source variables supported for ceda command"

    for year in years:

        # run create variable
        create_variable(
            config_paths=variable_configs,
            year=year,
            domain=domain,
            scale_factor=scale_factor,
            ensemble_member=ensemble_member,
            scenario=scenario,
            thetas=thetas,
            target_resolution=target_resolution,
        )


if __name__ == "__main__":
    app()
