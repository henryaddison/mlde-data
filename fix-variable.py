import os
import re

import iris
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

domain_var_resolutions = {
    "london": {
        "pr": [
            "2.2km-2.2km",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x"
        ],
        "vorticity850": [
            "2.2km-2.2km",
            "2.2km-coarsened-8x-2.2km",
            "2.2km-coarsened-gcm-2.2km",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ]
    },
    "birmingham": {
        "pr": [
            # "2.2km-2.2km",
            "2.2km-coarsened-4x-2.2km-coarsened-4x",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ],
        "vorticity850": [
            # "2.2km-2.2km",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
            "60km-2.2km",
            "60km-2.2km-coarsened-4x",
        ]
    }
}

years = list(range(1981, 2001))+list(range(2021, 2041))+list(range(2061, 2081))

def remove_forecast(ds):
    vars_to_remove = []
    coord_encodings_to_remove = []
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (re.match("(realization|forecast_period|forecast_reference_time) ?", ds[v].encoding["coordinates"]) is not None):
            coord_encodings_to_remove.append(v)
        if v in ["forecast_period", "forecast_reference_time", "realization", "forecast_period_bnds"]:
            vars_to_remove.append(v)

    if len(coord_encodings_to_remove) > 0:
        print("Dropping coordinates in encoding of", coord_encodings_to_remove)
    if len(vars_to_remove) > 0:
        print("Dropping vars", vars_to_remove)

    ds = ds.drop_vars(vars_to_remove)
    for v in ds.variables:
        if "coordinates" in ds[v].encoding:
            new_coords_encoding = re.sub("(realization|forecast_period|forecast_reference_time) ?", "", ds[v].encoding["coordinates"]).strip()
            ds[v].encoding.update({"coordinates": new_coords_encoding})

    return ds

for domain, var_resolutions in domain_var_resolutions.items():
    for var, resolutions in var_resolutions.items():
        for res in resolutions:
            ds_meta = UKCPDatasetMetadata(os.getenv("MOOSE_DERIVED_DATA"), variable=var, frequency="day", domain=domain, resolution=res)
            for year in range(1981, 2001):
                path = ds_meta.filepath(year)
                ds = xr.load_dataset(path)

                ds = remove_forecast(ds)

                ds.to_netcdf(path)
                iris.load(path)
