import logging
import os
from pathlib import Path

import typer
import xarray as xr

from mlde_utils import VariableMetadata
from .options import DomainOption
from mlde_utils.data.coarsen import Coarsen
from mlde_utils.data.select_domain import SelectDomain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def select_subdomain(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    subdomain: DomainOption = DomainOption.london,
    frequency: str = "day",
    resolution: str = "2.2km",
    domain: str = "uk",
    input_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    output_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
):
    """Select a subdomain within a given dataset"""
    scenario = "rcp85"
    ensemble_member = "01"

    input_ds_params = dict(
        domain=domain,
        frequency=frequency,
        variable=variable,
        ensemble_member=ensemble_member,
        scenario=scenario,
        resolution=resolution,
    )
    input = VariableMetadata(input_base_dir, **input_ds_params)

    output_ds_params = input_ds_params.copy()
    output_ds_params.update({"domain": subdomain.value})
    output = VariableMetadata(output_base_dir, **output_ds_params)

    typer.echo(f"Opening {input.filepath(year)}")
    ds = xr.open_dataset(input.filepath(year))
    typer.echo(f"Select {subdomain.value} subdomain...")
    ds = SelectDomain(subdomain=subdomain.value).run(ds)

    typer.echo(f"Saving to {output.filepath(year)}...")
    os.makedirs(output.dirpath(), exist_ok=True)
    ds.to_netcdf(output.filepath(year))


@app.command()
def coarsen(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    scale_factor: int = typer.Option(...),
    frequency: str = "day",
    resolution: str = "2.2km",
    domain: str = "uk",
    input_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    output_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
):
    """Coarsen provided dataset"""
    scenario = "rcp85"
    ensemble_member = "01"

    input_ds_params = dict(
        domain=domain,
        frequency=frequency,
        variable=variable,
        ensemble_member=ensemble_member,
        scenario=scenario,
        resolution=resolution,
    )
    input = VariableMetadata(input_base_dir, **input_ds_params)

    new_resolution = f"{resolution}-coarsened-{scale_factor}x"
    output_ds_params = input_ds_params.copy()
    output_ds_params.update({"resolution": new_resolution})
    output = VariableMetadata(output_base_dir, **output_ds_params)

    typer.echo(f"Opening {input.filepath(year)}")
    ds = xr.open_dataset(input.filepath(year))

    typer.echo(f"Coarsening {scale_factor}x...")
    ds = Coarsen(scale_factor=scale_factor, variable=variable).run(ds)

    typer.echo(f"Saving to {output.filepath(year)}...")
    os.makedirs(output.dirpath(), exist_ok=True)
    ds.to_netcdf(output.filepath(year))
