from argparse import ArgumentError
import logging
from pathlib import Path
import subprocess
from typing import List
import yaml

import typer

from ml_downscaling_emulator.bin import DomainOption, CollectionOption
from ml_downscaling_emulator.bin.moose import processed_nc_filepath, create_variable, extract, convert, clean

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

def run_cmd(cmd):
    logger.debug(f"Running {cmd}")
    output = subprocess.run(cmd, capture_output=True, check=False)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    output.check_returncode()

@app.command()
def main(years: List[int], variable_config: Path = typer.Option(...), domain: DomainOption = DomainOption.london, frequency: str = "day", scenario="rcp85", scale_factor: str = typer.Option(...), target_resolution: str = "2.2km", target_size: int = 64):

    with open(variable_config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    for year in years:
        # run extract and convert
        src_collection = config["sources"]["collection"]
        for src_variable in config["sources"]["variables"]:
            extract(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

            convert(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

        # run create variable
        create_variable(config_path=variable_config, year=year, domain=domain, target_resolution=target_resolution, target_size=target_size, scale_factor=scale_factor)

        # run transfer
        if src_collection == "land-cpm":
            src_resolution = "2.2km"
        elif src_collection == "land-gcm":
            src_resolution = "60km"
        else:
            raise(ArgumentError(f"Unknown source collection {src_collection}"))

        if scale_factor == "gcm":
            variable_resolution = f"{src_resolution}-coarsened-gcm"
        elif scale_factor == "1":
            variable_resolution = f"{src_resolution}"
        else:
            variable_resolution = f"{src_resolution}-coarsened-{scale_factor}x"

        jasmin_filepath = processed_nc_filepath(variable=config["variable"], year=year, frequency=frequency, domain=domain.str, resolution=f"{variable_resolution}-{target_resolution}")
        bp_filepath = processed_nc_filepath(variable=config["variable"], base_dir="/user/work/vf20964")

        file_xfer_cmd = ["~/code/ml-downscaling-emulation/cpm/hum/xfer-script", jasmin_filepath, bp_filepath]
        config_xfer_cmd = []
        run_cmd(file_xfer_cmd)

        # run clean up
        for src_variable in config["sources"]["variables"]:
            clean(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], collection=src_collection)

if __name__ == "__main__":
    app()
