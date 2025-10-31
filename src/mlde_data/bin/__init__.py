from dotenv import load_dotenv
import logging
import os
from pathlib import Path
import typer
from typing import List
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
def sample(files: List[Path], output_dir: Path = None, dim: str = "time"):
    for file in files:
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

        sampled_ds = sampled_ds

        ds.close()
        del ds

        if output_dir is not None:
            output_file = Path(output_dir) / Path(file).relative_to(Path(file).anchor)
            os.makedirs(output_file.parent, exist_ok=True)
            output_file = str(output_file)
        else:
            output_file = file
        print(f"Saving {output_file}")

        sampled_ds.to_netcdf(output_file)
        del sampled_ds


if __name__ == "__main__":
    app()
