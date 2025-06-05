from dotenv import load_dotenv
import logging
import os
from pathlib import Path
import typer
from typing import List
import xarray as xr

from . import ceda
from . import dataset
from . import moose
from . import preprocess
from . import variable

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger.setLevel(log_level)

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(variable.app, name="variable")


@app.command()
def sample(files: List[Path], output_dir: Path = None, dim: str = "time"):
    for file in files:
        ds = xr.open_dataset(file)
        # take something from first day of each month
        time_mask = ds[f"{dim}.dayofyear"] % 30 == 0

        sampled_ds = ds.sel({dim: time_mask}).load()

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
