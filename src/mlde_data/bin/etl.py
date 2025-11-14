import logging
import os
from pathlib import Path
from mlde_data import MOOSE_VARIABLES_PATH
from mlde_data.bin.options import DomainOption, CollectionOption
from mlde_data.bin.moose import extract, convert, clean
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
    variable_config: Path = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    size: int = typer.Option(...),
    theta: int = None,
    target_resolution: str = None,
    force: bool = False,
    cleanup: bool = True,
):

    config = load_config(
        variable_config,
        scale_factor=scale_factor,
        domain=domain.value,
        theta=theta,
        target_resolution=target_resolution,
    )

    for year in years:
        # run extract and convert
        src_collection = CollectionOption(config["sources"]["collection"])
        src_frequency = config["sources"]["frequency"]
        for src_variable in config["sources"]["variables"]:
            if src_collection == CollectionOption.cpm:
                source_domain = "uk"
                source_resolution = "2.2km"
            elif src_collection == CollectionOption.gcm:
                source_domain = "global"
                source_resolution = "60km"
            else:
                raise f"Unknown collection {src_collection}"
            source_nc_filepath = VariableMetadata(
                base_dir=MOOSE_VARIABLES_PATH,
                variable=src_variable["name"],
                frequency=src_frequency,
                domain=source_domain,
                resolution=source_resolution,
                ensemble_member=ensemble_member,
                scenario=scenario,
                collection=src_collection.value,
            ).filepath(year)
            # skip extract and convert if file already exists and not forcing an extraction
            if os.path.exists(source_nc_filepath) and not force:
                logger.info(f"{source_nc_filepath} already exists, skipping extraction")
                continue

            extract(
                variable=src_variable["name"],
                year=year,
                frequency=src_frequency,
                collection=src_collection,
                ensemble_member=ensemble_member,
                scenario=scenario,
            )
            convert(
                variable=src_variable["name"],
                year=year,
                frequency=src_frequency,
                collection=src_collection,
                ensemble_member=ensemble_member,
                scenario=scenario,
            )

        # run create variable
        create_variable(
            config_path=variable_config,
            year=year,
            domain=domain,
            size=size,
            scale_factor=scale_factor,
            ensemble_member=ensemble_member,
            scenario=scenario,
            theta=theta,
            target_resolution=target_resolution,
        )

        # run clean up
        if cleanup:
            for src_variable in config["sources"]["variables"]:
                clean(
                    collection=src_collection,
                    scenario=scenario,
                    ensemble_member=ensemble_member,
                    variable=src_variable["name"],
                    frequency=src_frequency,
                    year=year,
                )


if __name__ == "__main__":
    app()
