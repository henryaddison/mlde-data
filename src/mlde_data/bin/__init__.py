from pathlib import Path
from typing import List

import typer
import xarray as xr

from . import ceda
from . import dataset
from . import evaluate
from . import moose
from . import preprocess
from . import variable

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(variable.app, name="variable")


@app.command()
def sample(files: List[Path]):
    for file in files:
        ds = xr.open_dataset(file)
        # take something from each season and each decade
        sampled_ds = ds.sel(
            time=((ds["time.month"] % 3 == 0) & (ds["time.year"] % 10 == 0))
        ).load()
        ds.close()
        del ds
        print(f"Saving {file}")
        sampled_ds.to_netcdf(file)
        del sampled_ds


if __name__ == "__main__":
    app()
