import os

import iris
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

domain = "birmingham"
var="vorticity850"
resolutions = ["2.2km-coarsened-gcm-2.2km", "2.2km-coarsened-gcm-2.2km-coarsened-4x"]

for res in resolutions:
    ds_meta = UKCPDatasetMetadata(os.getenv("MOOSE_DERIVED_DATA"), variable=var, frequency="day", domain=domain, resolution=res)
    for year in range(1981, 2001):
        path = ds_meta.filepath(year)
        ds = xr.load_dataset(path)

        ds = ds.reset_coords(["forecast_reference_time", "realization"], drop=True)
        for v in ds.variables:
            print(v, ds[v].encoding.pop("coordinates", None))

        ds.to_netcdf(path)
        iris.load(path)
