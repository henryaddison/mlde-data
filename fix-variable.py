import os

import iris
import typer
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.bin.moose import remove_forecast

app = typer.Typer()

YEARS = list(range(1981, 2001))+list(range(2021, 2041))+list(range(2061, 2081))

@app.command()
def main(domain: str, res: str, var: str):
    ds_meta = UKCPDatasetMetadata(os.getenv("MOOSE_DERIVED_DATA"), variable=var, frequency="day", domain=domain, resolution=res)
    for year in YEARS:
        path = ds_meta.filepath(year)
        ds = xr.load_dataset(path)

        ds = remove_forecast(ds)

        ds.to_netcdf(path)
        iris.load(path)

if __name__ == "__main__":
    app()
