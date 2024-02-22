from importlib_resources import files
import logging
import os
import yaml

import typer
import xarray as xr

from mlde_utils import VariableMetadata
from .options import DomainOption
from mlde_utils.data.sum import Sum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.command()
def create_variable(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    resolution: str = typer.Option(...),
    frequency: str = "day",
    domain: DomainOption = DomainOption.london,
    scenario="rcp85",
):
    """
    Create a new variable
    """
    config = (
        files("mlde_data.config").joinpath(f"variables/day/{variable}.yml").read_text()
    )
    config = yaml.safe_load(config)

    data_basedir = os.path.join(os.getenv("DERIVED_DATA"), "moose")

    output_metadata = VariableMetadata(
        data_basedir,
        frequency=frequency,
        domain=domain,
        resolution=resolution,
        ensemble_member="01",
        variable=config["variable"],
    )

    for job_spec in config["spec"]:
        if job_spec["action"] == "sum":
            logger.info(f"Summing {job_spec['variables']}")
            input_metadata = [
                VariableMetadata(
                    data_basedir,
                    frequency=frequency,
                    domain=domain,
                    resolution=resolution,
                    ensemble_member="01",
                    variable=variable,
                )
                for variable in job_spec["variables"]
            ]

            ds = xr.open_mfdataset([m.filepath(year) for m in input_metadata])
            ds = Sum(
                [m.variable for m in input_metadata], output_metadata.variable
            ).run(ds)
            ds[variable] = ds[variable].assign_attrs(config["attrs"])

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)
    ds.to_netcdf(output_metadata.filepath(year))
