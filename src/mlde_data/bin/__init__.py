import os
from pathlib import Path
from typing import List

import typer
import xarray as xr

from . import ceda
from . import dataset
from . import moose
from . import preprocess
from . import variable

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(variable.app, name="variable")


@app.command()
def sample(files: List[Path], output_dir: Path = None):
    for file in files:
        ds = xr.open_dataset(file)
        # take something from each season and each decade
        time_mask = (ds["time.month"] % 3 == 0) & (ds["time.year"] % 10 == 0)
        # if empty mask then assume a small set and allow all years
        if not time_mask.any().item():
            time_mask = ds["time.month"] % 3 == 0

        sampled_ds = ds.sel(time=time_mask).load()

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
