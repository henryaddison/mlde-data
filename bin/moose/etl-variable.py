import logging
from pathlib import Path
import typer
from typing import List
import yaml

from mlde_data.bin.options import DomainOption, CollectionOption
from mlde_data.bin.moose import extract, convert, clean
from mlde_data.bin.variable import create as create_variable

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
):

    with open(variable_config, "r") as config_file:
        config = yaml.safe_load(config_file)

    for year in years:
        # run extract and convert
        src_collection = CollectionOption(config["sources"]["collection"])
        for src_variable in config["sources"]["variables"]:
            extract(
                variable=src_variable["name"],
                year=year,
                frequency=src_variable["frequency"],
                collection=src_collection,
                ensemble_member=ensemble_member,
                scenario=scenario,
            )

            convert(
                variable=src_variable["name"],
                year=year,
                frequency=src_variable["frequency"],
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
        )

        # run clean up
        for src_variable in config["sources"]["variables"]:
            clean(
                variable=src_variable["name"],
                year=year,
                frequency=src_variable["frequency"],
                collection=src_collection,
                ensemble_member=ensemble_member,
            )


if __name__ == "__main__":
    app()
