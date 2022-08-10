import os
import re

import iris
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

def remove_forecast(ds):
    vars_to_remove = []
    coord_encodings_to_remove = []
    for v in ds.variables:
        if ("coordinates" in ds[v].encoding) and (re.match("(realization|forecast_period|forecast_reference_time) ?", ds[v].encoding["coordinates"]) is not None):
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

datasets = [
    "2.2km-coarsened-8x_london_vorticity850_random",
    "2.2km-coarsened-gcm-2.2km-coarsened-4x_birmingham_vorticity850_random",
    "60km-2.2km-coarsened-4x_birmingham_vorticity850_random",
    "2.2km-coarsened-gcm-2.2km_london_vorticity850_random",
    "60km-2.2km_london_vorticity850_random"
]

splits = ["train", "val", "test"]

for dataset in datasets:
    for split in splits:
        print(f"Fixing {split} of {dataset}")
        dataset_path = os.path.join(os.getenv("MOOSE_DERIVED_DATA"), "nc-datasets", dataset, f"{split}.nc")
        ds = xr.load_dataset(dataset_path)
        # ds.close()
        ds = remove_forecast(ds)

        ds.to_netcdf(dataset_path)
