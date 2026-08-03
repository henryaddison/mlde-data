from collections import defaultdict
from mlde_utils import FurflexDatasetMetadata
import re
import xarray as xr


def validate(dataset: str) -> defaultdict:
    """
    Validate a dataset
    """
    splits = ["train", "val", "test"]

    bad_splits = defaultdict(set)

    dataset_metadata = FurflexDatasetMetadata(dataset)

    try:
        ds_config = dataset_metadata.config()
    except FileNotFoundError:
        bad_splits["no config"].update(splits)
        return bad_splits

    for split in dataset_metadata.splits():
        for var_type in ["predictors", "predictands"]:
            try:
                ds = xr.open_dataset(
                    dataset_metadata.split_path(split) / f"{var_type}.zarr"
                )
            except FileNotFoundError:
                bad_splits["no file"].add(split)
                continue

            if not check_grid_mapping(ds, dataset, split, var_type, ds_config):
                bad_splits["bad grid mapping"].add(split)

            # check dims
            if not check_dims(ds, dataset, split, var_type, ds_config):
                bad_splits["bad dimensions"].add(split)

            # check grid and time
            if not check_grid_vars(ds, dataset, split, var_type, ds_config):
                bad_splits["bad grid vars"].add(split)
            if not check_time_bnds(ds, dataset, split, var_type, ds_config):
                bad_splits["bad time_bnds"].add(split)
            if not check_time_encoding(ds, dataset, split, var_type, ds_config):
                bad_splits["bad time encodings"].add(split)

            # check shape
            if not check_shape(ds, dataset, split, var_type, ds_config):
                bad_splits["bad shape"].add(split)

            # check for forecast related metadata (should have been stripped)
            if not check_forecast_encoding(ds, dataset, split, var_type, ds_config):
                bad_splits["forecast_encoding"].add(split)
            if not check_forecast_variables(ds, dataset, split, var_type, ds_config):
                bad_splits["forecast_vars"].add(split)

            # check for pressure related metadata (should have been stripped)
            if not check_pressure_encoding(ds, dataset, split, var_type, ds_config):
                bad_splits["pressure_encoding"].add(split)
            if not check_pressure_variables(ds, dataset, split, var_type, ds_config):
                bad_splits["pressure_vars"].add(split)

            # check for NaNs
            if not check_nans(ds, dataset, split, var_type, ds_config):
                bad_splits["NaNs"].add(split)

    return bad_splits


def check_grid_mapping(ds, dataset, split, var_type, ds_config):
    grid_mappings = {
        ds[v].attrs["grid_mapping"] for v in ds_config[var_type]["variables"]
    }

    if len(grid_mappings) != 1:
        return False
    if not (
        grid_mappings
        <= {"rotated_latitude_longitude", "latitude_longitude", "transverse_mercator"}
    ):
        return False

    return True


def check_dims(ds, dataset, split, var_type, ds_config):
    example_var = f"{ds_config[var_type]['variables'][0]}"
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        return list(ds[example_var].dims) == [
            "ensemble_member",
            "time",
            "grid_latitude",
            "grid_longitude",
        ]
    elif grid_mapping == "transverse_mercator":
        return list(ds[example_var].dims) == [
            "ensemble_member",
            "time",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
    elif grid_mapping == "latitude_longitude":
        return list(ds[example_var].dims) == [
            "ensemble_member",
            "time",
            "latitude",
            "longitude",
        ]
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")


def check_shape(ds, dataset, split, var_type, ds_config):
    ems = ds_config["ensemble_members"]
    example_var = ds_config[var_type]["variables"][0]
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    if grid_mapping == "rotated_latitude_longitude":
        size = 64
    elif grid_mapping == "transverse_mercator":
        size = 128
    elif grid_mapping == "latitude_longitude":
        domain = ds.attrs.get("domain")
        if domain == "engwales":
            size = 13
        elif domain == "engwales-5km":
            size = 14
    else:
        raise RuntimeError(f"Unknown grid_mapping {grid_mapping}")

    expected_shape = [len(ems), size, size]
    actual_shape = list(ds[example_var].shape)
    # remove time dimension from actual shape as this is dataset dependent
    ntimes = actual_shape.pop(1)

    if split == "train":
        expected_ntimes = {15120, 25200}
        if var_type == "predictands":
            expected_ntimes = {24 * n for n in expected_ntimes}
    else:
        expected_ntimes = {3240, 5400}
        if var_type == "predictands":
            expected_ntimes = {24 * n for n in expected_ntimes}

    return (actual_shape == expected_shape) and (ntimes in expected_ntimes)


def check_grid_vars(ds, dataset, split, var_type, ds_config):
    example_var = ds_config[var_type]["variables"][0]
    grid_mapping = ds[example_var].attrs["grid_mapping"]
    meta_vars = [
        grid_mapping,
    ]
    if grid_mapping == "rotated_latitude_longitude":
        meta_vars.extend(["grid_latitude_bnds", "grid_longitude_bnds"])
    if grid_mapping == "transverse_mercator":
        meta_vars.extend(
            ["projection_x_coordinate_bnds", "projection_y_coordinate_bnds"]
        )
    if grid_mapping == "latitude_longitude":
        meta_vars.extend(["latitude_bnds", "longitude_bnds"])

    return all(
        [
            ("ensemble_member" not in ds[mvar].dims) and ("time" not in ds[mvar].dims)
            for mvar in meta_vars
        ]
    )


def check_time_bnds(ds, dataset, split, var_type, ds_config):
    return "time_bnds" in ds and "ensemble_member" not in ds["time_bnds"].dims


def check_forecast_encoding(ds, dataset, split, var_type, ds_config):
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match(
                "(realization|forecast_period|forecast_reference_time) ?",
                ds[v].encoding["coordinates"],
            )
            is not None
        ):
            return False
    return True


def check_forecast_variables(ds, dataset, split, var_type, ds_config):
    for v in ds.variables:
        if v in [
            "forecast_period",
            "forecast_reference_time",
            "realization",
            "forecast_period_bnds",
        ]:
            return False
    return True


def check_pressure_encoding(ds, dataset, split, var_type, ds_config):
    for v in ds.variables:
        if "coordinates" in ds[v].encoding and (
            re.match("(pressure) ?", ds[v].encoding["coordinates"]) is not None
        ):
            return False
    return True


def check_pressure_variables(ds, dataset, split, var_type, ds_config):
    for v in ds.variables:
        if v in ["pressure"]:
            return False
    return True


def check_nans(ds, dataset, split, var_type, ds_config):
    for v in ds.variables:
        nan_count = ds[v].isnull().sum().values.item()
        if nan_count > 0:
            return False
    return True


def check_time_encoding(ds, dataset, split, var_type, ds_config):
    time_encodings = [ds.time.encoding]
    if "time_bnds" in ds:
        time_encodings.append(ds.time_bnds.encoding)
    for enc in time_encodings:
        if enc["units"] not in [
            "hours since 1970-01-01",
            "microseconds since 1970-01-01",
        ]:
            return False
        if enc["calendar"] != "360_day":
            return False

    return True
