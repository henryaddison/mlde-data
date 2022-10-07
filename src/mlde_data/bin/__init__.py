from enum import Enum
from pathlib import Path
from typing import List

import typer
import xarray as xr

class DomainOption(str, Enum):
    uk = "uk"
    london = "london"
    birmingham = "birmingham"

class CollectionOption(str, Enum):
    gcm = "land-gcm"
    cpm = "land-cpm"

from . import ceda
from . import dataset
from . import evaluation
from . import moose
from . import preprocess
from . import variable

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(evaluation.app, name="evaluate")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(variable.app, name="variable")

@app.command()
def sample(files: List[Path]):
    for file in files:
        ds = xr.open_dataset(file)
        sampled_ds = ds.isel(time=slice(100)).load()
        ds.close()
        del ds
        print(f"Saving {file}")
        sampled_ds.to_netcdf(file)
        del sampled_ds

if __name__ == "__main__":
    app()