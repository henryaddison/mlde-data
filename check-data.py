import os

import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata

domain_var_resolutions = {
    "london": {
        "pr": [
            "2.2km-2.2km",
            "2.2km-coarsened-2x-2.2km-coarsened-2x",
            "2.2km-coarsened-4x-2.2km-coarsened-4x",
            "2.2km-coarsened-8x-2.2km-coarsened-8x",
            "2.2km-coarsened-27x-2.2km",
            "2.2km-coarsened-27x-2.2km-coarsened-2x",
            "2.2km-coarsened-27x-2.2km-coarsened-4x",
            "2.2km-coarsened-27x-2.2km-coarsened-8x",

        ],
        "vorticity850": [
            "2.2km-2.2km",
            "2.2km-coarsened-4x-2.2km",
            "2.2km-coarsened-8x-2.2km",
            "2.2km-coarsened-27x-2.2km",
            "2.2km-coarsened-27x-2.2km-coarsened-2x",
            "2.2km-coarsened-27x-2.2km-coarsened-4x",
            "2.2km-coarsened-27x-2.2km-coarsened-8x",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-2x",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
            "2.2km-coarsened-gcm-2.2km-coarsened-8x",
            "60km-2.2km",
            "60km-2.2km-coarsened-2x",
            "60km-2.2km-coarsened-4x",
            "60km-2.2km-coarsened-8x",
        ]
    },
    "birmingham": {
        "pr": [
            "2.2km-2.2km",
            "2.2km-coarsened-8x-2.2km",
            "2.2km-coarsened-8x-2.2km-coarsened-4x",
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
        ],
        "vorticity850": [
            "2.2km-coarsened-gcm-2.2km",
            "2.2km-coarsened-gcm-2.2km-coarsened-4x",
        ]
    }
}

years = list(range(1981, 2001))+list(range(2021, 2041))+list(range(2061, 2081))

for domain, var_resolutions in domain_var_resolutions.items():
    for var, resolutions in var_resolutions.items():
        for res in resolutions:
            for year in years:
                var_meta = UKCPDatasetMetadata(os.getenv("MOOSE_DERIVED_DATA"), variable=var, frequency="day", domain=domain, resolution=res)

                nan_count = xr.open_dataset(var_meta.filepath(year))[var].isnull().sum().values.item()

                try:
                    assert nan_count == 0
                    print(f"Checking {year} for {var} over {domain} at {res}")
                except AssertionError:
                    print(f"Failed {year} for {var} over {domain} at {res}: {nan_count} NaNs")

