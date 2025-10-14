from dotenv import load_dotenv
import logging
import numpy as np
import os
from pathlib import Path
import typer
import xarray as xr

from . import dataset, etl, moose, variable

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger.setLevel(log_level)

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(dataset.app, name="dataset")
app.add_typer(etl.app, name="etl")
app.add_typer(moose.app, name="moose")
app.add_typer(variable.app, name="variable")


@app.command()
def sample(file: Path, output_file: Path, dim: str = "time"):
    ds = xr.open_dataset(file)

    # take something from first day of each mont
    doy_mask = ds[f"{dim}.dayofyear"] % 30 == 0
    sampled_ds = ds.sel({dim: doy_mask}).load()

    # if covers a long time period, take only specific years
    year_mask = (ds["time.year"] + (ds["time.month"] == 12)).isin([1981,2000,2021,2040,2061,2080])
    if np.any(year_mask):
        sampled_ds = sampled_ds.sel({dim: year_mask})

    # if ensemble member dimension exists, take only first 3 members
    if "ensemble_member" in sampled_ds.dims and len(sampled_ds["ensemble_member"]) >= 3:
        sampled_ds = sampled_ds.isel(ensemble_member=[0,1,2])
    ds.close()

    # ensure output directory exists
    os.makedirs(output_file.parent, exist_ok=True)
    output_file = str(output_file)

    logger.info(f"Saving {output_file}")
    sampled_ds.to_netcdf(output_file)


if __name__ == "__main__":
    app()
