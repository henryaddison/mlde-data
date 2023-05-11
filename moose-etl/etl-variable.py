from argparse import ArgumentError
import logging
from pathlib import Path
import random
from typing import List
import time
import yaml

import typer

from mlde_data.bin.options import DomainOption, CollectionOption
from mlde_data.bin.moose import extract, convert, clean
from mlde_data.bin.variable import create as create_variable, xfer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.command()
def main(
    years: List[int],
    variable_config: Path = typer.Option(...),
    ensemble_member: str = typer.Option(...),
    domain: DomainOption = typer.Option(...),
    frequency: str = typer.Option(...),
    scale_factor: str = typer.Option(...),
    target_resolution: str = typer.Option(...),
    scenario: str = "rcp85",
    target_size: int = 64,
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
                cache=False,
            )

            convert(
                variable=src_variable["name"],
                year=year,
                frequency=src_variable["frequency"],
                collection=src_collection,
                ensemble_member=ensemble_member,
                cache=False,
            )

        # run create variable
        create_variable(
            config_path=variable_config,
            year=year,
            domain=domain,
            target_resolution=target_resolution,
            target_size=target_size,
            scale_factor=scale_factor,
            ensemble_member=ensemble_member,
        )

        # run transfer
        if src_collection == CollectionOption.cpm:
            src_resolution = "2.2km"
        elif src_collection == CollectionOption.gcm:
            src_resolution = "60km"
        else:
            raise (ArgumentError(f"Unknown source collection {src_collection}"))

        if scale_factor == "gcm":
            variable_resolution = f"{src_resolution}-coarsened-gcm"
        elif scale_factor == "1":
            variable_resolution = f"{src_resolution}"
        else:
            variable_resolution = f"{src_resolution}-coarsened-{scale_factor}x"

        resolution = f"{variable_resolution}-{target_resolution}"
        MAX_ATTEMPTS = 3
        attempts = 0
        while True:
            attempts += 1
            try:
                xfer(
                    variable=config["variable"],
                    year=year,
                    frequency=frequency,
                    domain=domain,
                    collection=src_collection,
                    resolution=resolution,
                    target_size=target_size,
                    ensemble_member=ensemble_member,
                )
            except Exception:
                if attempts >= MAX_ATTEMPTS:
                    raise
                else:
                    # pause for a bit
                    sleep_duration = 60 * (2 ** (attempts - 1)) + random.randint(1, 10)
                    logger.error(
                        f"Failed to do xfer on attempt {attempts} of {MAX_ATTEMPTS}. Sleeping for {sleep_duration}."
                    )
                    time.sleep(sleep_duration)

                    continue
            else:
                break

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
