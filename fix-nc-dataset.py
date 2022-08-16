import os

import iris
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.bin.moose import remove_forecast

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
