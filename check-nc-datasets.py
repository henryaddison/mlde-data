import os
import re

import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

datasets = [
    "2.2km-coarsened-8x_london_vorticity850_random",
    "2.2km-coarsened-gcm-2.2km-coarsened-4x_birmingham_vorticity850_random",
    "60km-2.2km-coarsened-4x_birmingham_vorticity850_random",
    "2.2km-coarsened-gcm-2.2km_london_vorticity850_random",
    "60km-2.2km_london_vorticity850_random"
]

splits = ["train", "val", "test"]

for dataset in datasets:
    bad_splits = {"no file": set(), "forecast_encoding": set(), "forecast_vars": set()}
    for split in splits:
        print(f"Checking {split} of {dataset}")
        dataset_path = os.path.join(os.getenv("MOOSE_DERIVED_DATA"), "nc-datasets", dataset, f"{split}.nc")
        try:
            ds = xr.open_dataset(dataset_path)
        except FileNotFoundError:
            bad_splits["no file"].add(split)
            continue

        # check for forecast related metadata (should have been stripped)
        for v in ds.variables:
            if "coordinates" in ds[v].encoding and (re.match("(realization|forecast_period|forecast_reference_time) ?", ds[v].encoding["coordinates"]) is not None):
                bad_splits["forecast_encoding"].add(split)
            if v in ["forecast_period", "forecast_reference_time", "realization", "forecast_period_bnds"]:
                bad_splits["forecast_vars"].add(split)

    # report findings
    for reason, error_splits in bad_splits.items():
        if len(error_splits) > 0:
            print(f"Failed '{reason}': {dataset} for {error_splits}")
