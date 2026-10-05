"""
Helper functions for validating variables
"""

import cf_xarray  # noqa: F401
import re
from typing import List
import xarray as xr

from mlde_utils import VariableMetadata

from mlde_data import moose
from mlde_data.canari_le_sprint_variable_adapter import CanariLESprintVariableAdapter


DOMAIN_RES_VARS = {
    # "canari-le-sprint": {
    #     "canari-le-sprint": {
    #         "birmingham-64": {
    #             "60km-2.2km-coarsened-4x": [
    #                 "psl",
    #                 "pr",
    #                 "temp250",
    #                 "temp500",
    #                 "temp700",
    #                 "temp850",
    #                 "vorticity250",
    #                 "vorticity500",
    #                 "vorticity700",
    #                 "vorticity850",
    #             ],
    #         },
    #     },
    # },
    "moose": {
        "land-cpm": [
            {
                "domain": "birmingham-64",
                "resolution": "2.2km-coarsened-gcm-2.2km-coarsened-4x",
                "frequency": "day",
                "variables": [
                    "psl",
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
            },
            {
                "domain": "engwales",
                "resolution": "2.2km-coarsened-gcm",
                "frequency": "day",
                "variables": [
                    "psl",
                    "vorticity250",
                    "vorticity500",
                    "vorticity700",
                    "vorticity850",
                    "vort250",
                    "vort500",
                    "vort700",
                    "vort850",
                    "spechum250",
                    "spechum500",
                    "spechum700",
                    "spechum850",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                ],
            },
            {
                "domain": "engwales-5km",
                "resolution": "2.2km-coarsened-gcm",
                "frequency": "day",
                "variables": [
                    "psl",
                    "vort250",
                    "vort500",
                    "vort700",
                    "vort850",
                    "spechum250",
                    "spechum500",
                    "spechum700",
                    "spechum850",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                ],
            },
            {
                "domain": "uk",
                "resolution": "2.2km-coarsened-gcm",
                "frequency": "day",
                "variables": [
                    "psl",
                    "vort250",
                    "vort500",
                    "vort700",
                    "vort850",
                    "spechum250",
                    "spechum500",
                    "spechum700",
                    "spechum850",
                    "temp250",
                    "temp500",
                    "temp700",
                    "temp850",
                ],
            },
            {
                "domain": "engwales",
                "resolution": "2.2km-coarsened-4x",
                "frequency": "1hr",
                "variables": [
                    "pr",
                ],
            },
            {
                "domain": "engwales-5km",
                "resolution": "5km",
                "frequency": "1hr",
                "variables": [
                    "pr",
                ],
            },
            {
                "domain": "uk",
                "resolution": "5km",
                "frequency": "1hr",
                "variables": [
                    "pr",
                ],
            },
        ]
    },
}

YEARS = {
    "moose": list(range(1981, 2001))
    + list(range(2001, 2021))
    + list(range(2021, 2041))
    + list(range(2041, 2061))
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


def check_nans(ds: xr.Dataset, var: VariableMetadata) -> bool:
    return ds[var.variable].isnull().sum().values.item() == 0


def check_dims(ds: xr.Dataset, var: VariableMetadata) -> bool:
    grid_mapping = ds.cf[var.variable].grid_mapping
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[var.variable].dims) == [
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[var.variable].dims) == [
            "time",
            "latitude",
            "longitude",
        ]
    elif grid_mapping == "transverse_mercator":
        return list(ds[var.variable].dims) == [
            "time",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_time_shape(ds: xr.Dataset, var: VariableMetadata) -> bool:
    if var.frequency == "day":
        return len(ds[var.variable].cf["T"]) == 360
    elif var.frequency == "1hr":
        return len(ds[var.variable].cf["T"]) == 360 * 24
    else:
        raise RuntimeError(f"Unknown frequency {var.frequency}")


def check_spatial_shape(ds: xr.Dataset, var: VariableMetadata) -> bool:
    if var.domain == "engwales":
        if var.resolution in ["2.2km-coarsened-gcm", "60km"]:
            return (
                len(ds[var.variable].cf["X"]) == 13
                and len(ds[var.variable].cf["Y"]) == 13
            )
        elif var.resolution == "2.2km-coarsened-4x":
            return (
                len(ds[var.variable].cf["X"]) == 64
                and len(ds[var.variable].cf["Y"]) == 64
            )
    elif var.domain == "engwales-5km":
        if var.resolution == "5km":
            return (
                len(ds[var.variable].cf["X"]) == 13
                and len(ds[var.variable].cf["Y"]) == 13
            )
        elif var.resolution in ["2.2km-coarsened-gcm", "60km"]:
            return (
                len(ds[var.variable].cf["X"]) == 14
                and len(ds[var.variable].cf["Y"]) == 14
            )
        elif var.resolution == "5km":
            return (
                len(ds[var.variable].cf["X"]) == 128
                and len(ds[var.variable].cf["Y"]) == 128
            )
    elif var.domain == "uk":
        if var.resolution in ["2.2km-coarsened-gcm", "60km"]:
            return (
                len(ds[var.variable].cf["X"]) == 21
                and len(ds[var.variable].cf["Y"]) == 21
            )
        elif var.resolution == "5km":
            return (
                len(ds[var.variable].cf["X"]) == 180
                and len(ds[var.variable].cf["Y"]) == 244
            )

    raise RuntimeError(
        f"Unknown domain and resolution combination: {var.domain}, {var.resolution}"
    )


def check_forecast_encoding(ds: xr.Dataset, var: VariableMetadata) -> bool:
    if "coordinates" in ds[var.variable].encoding and (
        re.match(
            "(realization|forecast_period|forecast_reference_time) ?",
            ds[var.variable].encoding["coordinates"],
        )
        is not None
    ):
        return False
    return True


def check_forecast_vars(ds: xr.Dataset, var: VariableMetadata) -> bool:
    for v in ds.variables:
        if v in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "forecast_period_bnds",
        ]:
            return False
    return True


def check_pressure_encoding(ds: xr.Dataset, var: VariableMetadata) -> bool:
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
        ):
            return False
    return True


def check_pressure_vars(ds: xr.Dataset, var: VariableMetadata) -> bool:
    for v in ds.variables:
        if v in ["pressure"]:
            return False
    return True


def check_grid_vars(ds: xr.Dataset, var: VariableMetadata) -> bool:
    grid_mapping = ds[var.variable].attrs["grid_mapping"]
    meta_vars = [
        grid_mapping,
    ]
    if grid_mapping == "rotated_latitude_longitude":
        meta_vars.extend(["grid_latitude_bnds", "grid_longitude_bnds"])

    for mvar in meta_vars:
        if mvar not in ds.variables:
            return False
        if ("ensemble_member" in ds[mvar].dims) or ("time" in ds[mvar].dims):
            return False
    return True


def check_time_bnds(ds: xr.Dataset, var: VariableMetadata) -> bool:
    if "time_bnds" not in ds.variables:
        return False
    return "ensemble_member" not in ds["time_bnds"].dims


# currently these are missing from variable files
def check_spatial_bnds(ds: xr.Dataset, var: VariableMetadata) -> bool:
    grid_mapping = ds.cf[var.variable].grid_mapping
    if grid_mapping == "rotated_latitude_longitude":
        return (
            "grid_latitude_bnds" in ds.variables
            and "grid_longitude_bnds" in ds.variables
        )

    return True  # for other grid mappings, we don't currently expect spatial bounds, so return True


def check_variable_attrs(ds: xr.Dataset, var: VariableMetadata) -> bool:
    if var.resolution in ["2.2km-coarsened-gcm", "60km"]:
        expected_grid_mapping = "latitude_longitude"
    elif var.resolution in ["2.2km-coarsened-4x"]:
        expected_grid_mapping = "rotated_latitude_longitude"
    elif var.resolution in ["5km"]:
        expected_grid_mapping = "transverse_mercator"
    else:
        raise RuntimeError(f"Unknown resolution {var.resolution}")

    if var.variable == "pr":
        expected_attrs = {"units": "mm/hour", "standard_name": "lwe_precipitation_rate"}
    elif var.variable == "psl":
        expected_attrs = {"units": "Pa", "standard_name": "air_pressure_at_sea_level"}
    elif var.variable in ["temp250", "temp500", "temp700", "temp850"]:
        expected_attrs = {"units": "K", "standard_name": "air_temperature"}

    elif var.variable in ["spechum250", "spechum500", "spechum700", "spechum850"]:
        expected_attrs = {"units": 1, "standard_name": "specific_humidity"}
    elif var.variable in [
        "vorticity250",
        "vorticity500",
        "vorticity700",
        "vorticity850",
        "vort250",
        "vort500",
        "vort700",
        "vort850",
    ]:
        expected_attrs = {"units": "s-1", "standard_name": "relative_vorticity"}
    else:
        raise RuntimeError(f"Unknown variable {var.variable}")

    expected_attrs["grid_mapping"] = expected_grid_mapping

    return {
        k: ds[var.variable].attrs[k] for k in expected_attrs.keys()
    } == expected_attrs


def check_coord_attrs(ds: xr.Dataset, var: VariableMetadata) -> bool:
    expected_attrs = {
        "time": {"standard_name": "time", "bounds": "time_bnds", "axis": "T"}
    }

    grid_mapping = ds.cf[var.variable].grid_mapping
    if grid_mapping == "rotated_latitude_longitude":
        expected_attrs.update(
            {
                "grid_longitude": {
                    "standard_name": "grid_longitude",
                    "axis": "X",
                    "bounds": "grid_longitude_bnds",
                    "units": "degrees",
                },
                "grid_latitude": {
                    "standard_name": "grid_latitude",
                    "axis": "Y",
                    "bounds": "grid_latitude_bnds",
                    "units": "degrees",
                },
            }
        )
    elif grid_mapping == "latitude_longitude":
        expected_attrs.update(
            {
                "longitude": {
                    "standard_name": "longitude",
                    "axis": "X",
                    "long_name": "longitude",
                    "units": "degrees_east",
                },
                "latitude": {
                    "standard_name": "latitude",
                    "axis": "Y",
                    "long_name": "latitude",
                    "units": "degrees_north",
                },
            }
        )
    elif grid_mapping == "transverse_mercator":
        expected_attrs.update(
            {
                "projection_x_coordinate": {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "bounds": "projection_x_coordinate_bnds",
                    "units": "m",
                },
                "projection_y_coordinate": {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "bounds": "projection_y_coordinate_bnds",
                    "units": "m",
                },
            }
        )
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")

    actual_attrs = {k: ds[k].attrs for k in expected_attrs.keys()}

    return actual_attrs == expected_attrs


def check_ds_attrs(ds: xr.Dataset, var: VariableMetadata) -> bool:
    expected_attrs = {
        "domain": var.domain,
        "resolution": var.resolution,
        "frequency": var.frequency,
    }
    actual_attrs = {k: ds.attrs[k] for k in expected_attrs.keys()}

    return expected_attrs == actual_attrs


def validate(var_meta: VariableMetadata, year: int) -> List[str]:
    failures = []
    try:
        ds = xr.load_dataset(var_meta.filepath(year))
    except FileNotFoundError:
        failures.append("no file")
        return failures
    except Exception:
        failures.append("bad file")
        return failures

    # check for NaNs
    if not check_nans(ds, var_meta):
        failures.append("NaNs")

    # check dims
    if not check_dims(ds, var_meta):
        failures.append("dimensions")
    if not check_time_shape(ds, var_meta):
        failures.append("time shape")
    if not check_spatial_shape(ds, var_meta):
        failures.append("spatial shape")

    # check for forecast related metadata (should have been stripped)
    if not check_forecast_encoding(ds, var_meta):
        failures.append("forecast_encoding")
    if not check_forecast_vars(ds, var_meta):
        failures.append("forecast_vars")
    # check for pressure related metadata (should have been stripped)
    if not check_pressure_encoding(ds, var_meta):
        failures.append("pressure_encoding")
    if not check_pressure_vars(ds, var_meta):
        failures.append("pressure_vars")

    # check grid and time vars
    if not check_grid_vars(ds, var_meta):
        failures.append("grid meta vars")
    if not check_time_bnds(ds, var_meta):
        failures.append("time bnds")
    if not check_spatial_bnds(ds, var_meta):
        failures.append("spatial bnds")

    # check variable attributes
    if not check_variable_attrs(ds, var_meta):
        failures.append("variable attrs")
    if not check_coord_attrs(ds, var_meta):
        failures.append("coord attrs")
    if not check_ds_attrs(ds, var_meta):
        failures.append("ds attrs")
    return failures
