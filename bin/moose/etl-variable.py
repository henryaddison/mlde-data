import logging
from pathlib import Path
from mlde_data.bin.options import DomainOption, CollectionOption
from mlde_data.bin.moose import extract, convert, clean
from mlde_data.bin.variable import create as create_variable
from mlde_data.variable import load_config
import typer
from typing import List


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.command()
def main(
    years: List[int],
    variable_config: Path = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    scale_factor: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    size: int = typer.Option(...),
    theta: int = None,
    target_resolution: str = None,
):

    config = load_config(
        variable_config,
        scale_factor=scale_factor,
        domain=domain.value,
        size=size,
        theta=theta,
        target_resolution=target_resolution,
    )

    for year in years:
        # run extract and convert
        src_collection = CollectionOption(config["sources"]["collection"])
        src_frequency = config["sources"]["frequency"]
        for src_variable in config["sources"]["variables"]:
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
