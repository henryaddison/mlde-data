import logging
from pathlib import Path
from typing import List
import typer

from mlde_data.bin.options import DomainOption
from mlde_data.bin.variable import create as create_variable

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
    scenario: str = "ssp370",
    target_size: int = 64,
):

    for year in years:
        # run create variable
        create_variable(
            config_path=variable_config,
            year=year,
            domain=domain,
            target_resolution=target_resolution,
            target_size=target_size,
            scale_factor=scale_factor,
            ensemble_member=ensemble_member,
            scenario=scenario,
            frequency=frequency,
        )


if __name__ == "__main__":
    app()
