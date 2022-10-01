from argparse import ArgumentError
import logging
import os
from pathlib import Path
import subprocess
from typing import List
import yaml

import typer

from ml_downscaling_emulator.bin import DomainOption, CollectionOption
from ml_downscaling_emulator.bin.moose import xfer, create_variable, extract, convert, clean

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

@app.command()
def main(years: List[int], variable_config: Path = typer.Option(...), domain: DomainOption = DomainOption.london, frequency: str = "day", scenario="rcp85", scale_factor: str = typer.Option(...), target_resolution: str = "2.2km", target_size: int = 64):

    with open(variable_config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    for year in years:
        # run extract and convert
        src_collection = CollectionOption(config["sources"]["collection"])
        for src_variable in config["sources"]["variables"]:
            extract(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

            convert(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

        # run create variable
        create_variable(config_path=variable_config, year=year, domain=domain, target_resolution=target_resolution, target_size=target_size, scale_factor=scale_factor)

        # run transfer
        if src_collection == CollectionOption.cpm:
            src_resolution = "2.2km"
        elif src_collection == CollectionOption.gcm:
            src_resolution = "60km"
        else:
            raise(ArgumentError(f"Unknown source collection {src_collection}"))

        if scale_factor == "gcm":
            variable_resolution = f"{src_resolution}-coarsened-gcm"
        elif scale_factor == "1":
            variable_resolution = f"{src_resolution}"
        else:
            variable_resolution = f"{src_resolution}-coarsened-{scale_factor}x"

        resolution = f"{variable_resolution}-{target_resolution}"
        for attempt in reversed(range(3)):
            try:
                xfer(variable=config["variable"], year=year, frequency=frequency, domain=domain, collection=src_collection, resolution=resolution, target_size=target_size)
            except:
                if attempt <= 0:
                    raise
                else:
                    continue
            else:
                break

        # run clean up
        for src_variable in config["sources"]["variables"]:
            clean(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

if __name__ == "__main__":
    app()
