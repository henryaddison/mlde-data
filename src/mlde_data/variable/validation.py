"""
Helper functions for validating variables
"""

import re
from typing import List
import xarray as xr

from mlde_utils import VariableMetadata

from mlde_data import moose
from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter


DOMAIN_RES_VARS = {
    "canari-le-sprint": {
        "canari-le-sprint": {
            "birmingham-64": {
                "60km-2.2km-coarsened-4x": [
                    "psl",
                    "pr",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                    "vorticity250",
                    "vorticity500",
                    "vorticity700",
                    "vorticity850",
                ],
            },
        },
    },
    "moose": {
        "land-cpm": {
            "birmingham-64": {
                "2.2km-coarsened-gcm-2.2km-coarsened-4x": [
                    "psl",
                    # "tempgrad500250",
                    # "tempgrad700500",
                    # "tempgrad850700",
                    # "tempgrad925850",
                    "vorticity250",
                    "vorticity500",
                    "vorticity700",
                    "vorticity850",
                    "vorticity925",
                    "spechum250",
                    "spechum500",
                    "spechum700",
                    "spechum850",
                    "spechum925",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                    "temp925",
                    # "pr",
                    "linpr",
                ],
                "2.2km-coarsened-4x-2.2km-coarsened-4x": [
                    "pr",
                    "relhum150cm",
                    "tmean150cm",
                ],
            },
            "birmingham-9": {"2.2km-coarsened-gcm-60km": ["pr"]},
        },
        "land-gcm": {
            "birmingham-64": {
                "60km-2.2km-coarsened-4x": [
                    "psl",
                    # "tempgrad500250",
                    # "tempgrad700500",
                    # "tempgrad850700",
                    # "tempgrad925850",
                    "vorticity250",
                    "vorticity500",
                    "vorticity700",
                    "vorticity850",
                    "vorticity925",
                    "spechum250",
                    "spechum500",
                    "spechum700",
                    "spechum850",
                    "spechum925",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                    "temp925",
                    "linpr",
                    "pr",
                ],
            },
            "birmingham-9": {"60km-60km": ["pr"]},
        },
    },
}

YEARS = {
    "moose": list(range(1981, 2001))
    + list(range(2021, 2041))
    + list(range(2061, 2081)),
    "canari-le-sprint": list(range(1981, 1990)) + list(range(2071, 2080)),
}

ENSEMBLE_MEMBERS = {
    "moose": {
        "land-cpm": moose.SUITE_IDS["land-cpm"].keys(),
        "land-gcm": moose.RIP_CODES["land-gcm"].keys(),
    },
    "canari-le-sprint": {
        "canari-le-sprint": CanariLESprintVariableAdapter.ENSEMBLE_MEMBERS[
            CanariLESprintVariableAdapter.SSP370
        ].keys()
    },
}

SCENARIOS = {
    "moose": ["rcp85"],
    "canari-le-sprint": ["ssp370"],
}


def check_nans(ds: xr.Dataset, var: str) -> bool:
    return ds[var].isnull().sum().values.item() == 0


def check_dims(ds: xr.Dataset, var: str) -> bool:
    grid_mapping = ds[var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[var].dims) == [
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[var].dims) == [
            "time",
            "latitude",
            "longitude",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_shape(ds: xr.Dataset, var: str) -> bool:
    return len(ds[var]["time"]) == 360


def check_forecast_encoding(ds: xr.Dataset, var: str) -> bool:
    if "coordinates" in ds[var].encoding and (
        re.match(
            "(realization|forecast_period|forecast_reference_time) ?",
            ds[var].encoding["coordinates"],
        )
        is not None
    ):
        return False
    return True


def check_forecast_vars(ds: xr.Dataset, var: str) -> bool:
    for v in ds.variables:
        if v in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "forecast_period_bnds",
        ]:
            return False
    return True


def check_pressure_encoding(ds: xr.Dataset, var: str) -> bool:
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
        ):
            return False
    return True


def check_pressure_vars(ds: xr.Dataset, var: str) -> bool:
    for v in ds.variables:
        if v in ["pressure"]:
            return False
    return True


def check_grid_vars(ds: xr.Dataset, var: str) -> bool:
    grid_mapping = ds[var].attrs["grid_mapping"]
    meta_vars = [
        grid_mapping,
    ]
    if grid_mapping == "rotated_latitude_longitude":
        meta_vars.extend(["grid_latitude_bnds", "grid_longitude_bnds"])

    for mvar in meta_vars:
        if ("ensemble_member" in ds[mvar].dims) or ("time" in ds[mvar].dims):
            return False
    return True


def check_time_bnds(ds: xr.Dataset, var: str) -> bool:
    if "time_bnds" not in ds.variables:
        return False
    return "ensemble_member" not in ds["time_bnds"].dims


def validate(var_meta: VariableMetadata, year: int) -> List[str]:
    failures = []
    try:
        ds = xr.load_dataset(var_meta.filepath(year))
    except FileNotFoundError:
        failures.append("no file")
        return failures

    # check for NaNs
    if not check_nans(ds, var_meta.variable):
        failures.append("NaNs")

    # check dims
    if not check_dims(ds, var_meta.variable):
        failures.append("bad dimensions")
    if not check_shape(ds, var_meta.variable):
        failures.append("bad shape")

    # check for forecast related metadata (should have been stripped)
    if not check_forecast_encoding(ds, var_meta.variable):
        failures.append("forecast_encoding")
    if not check_forecast_vars(ds, var_meta.variable):
        failures.append("forecast_vars")
    # check for pressure related metadata (should have been stripped)
    if not check_pressure_encoding(ds, var_meta.variable):
        failures.append("pressure_encoding")
    if not check_pressure_vars(ds, var_meta.variable):
        failures.append("pressure_vars")
    # check grid and time vars
    if not check_grid_vars(ds, var_meta.variable):
        failures.append("grid_meta_vars")
    if not check_time_bnds(ds, var_meta.variable):
        failures.append("time_bnds")

    return failures
