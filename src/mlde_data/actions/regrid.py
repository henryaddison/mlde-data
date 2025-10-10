import cf_xarray  # noqa: F401
from importlib.resources import files
import iris
import iris.analysis
import logging
from mlde_data.actions.actions_registry import register_action
import xarray as xr

"""
Regrid a dataset based on a given target grid file
"""


@register_action(name="regrid_to_target")
class Regrid:

    SCHEMES = {
        "linear": iris.analysis.Linear,
        "nn": iris.analysis.Nearest,
        "area-weighted": iris.analysis.AreaWeighted,
    }

    def __init__(self, target_grid_resolution, variables, scheme="nn") -> None:
        self.target_grid_resolution = target_grid_resolution
        target_grid_filepath = files("mlde_data.actions").joinpath(
            f"target_grids/{self.target_grid_resolution}/uk/moose_grid.nc"
        )
        self.target_cube = iris.load_cube(target_grid_filepath)
        self.target_ds = xr.open_dataset(target_grid_filepath)
        self.variables = variables
        self.scheme = self.SCHEMES[scheme]()

    def __call__(self, ds):
        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only communicate in dataarrays not datasets
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data

        # if ds.attrs["resolution"] == self.target_grid_resolution:
        #     logging.debug("Already on the desired grid resolution, nothing to do")
        #     return ds

        logging.info(f"Regridding to target grid...")
        if ds.attrs["resolution"] == self.target_grid_resolution:
            logging.info("Already on the desired grid resolution, nothing to do")
            return ds

        src_coord_sys = self._source_coord_sys(ds)

        if "latitude_longitude" in self.target_ds.cf.grid_mapping_names:
            target_grid_mapping = "latitude_longitude"
        elif "rotated_latitude_longitude" in self.target_ds.cf.grid_mapping_names:
            target_grid_mapping = "rotated_latitude_longitude"
        elif "transverse_mercator" in self.target_ds.cf.grid_mapping_names:
            target_grid_mapping = "transverse_mercator"
        else:
            raise RuntimeError("Unrecognised grid system")

        vars = {}

        for variable in self.variables:
            src_cube = self._da_to_iris(ds[variable], src_coord_sys)

            regridder = self.scheme.regridder(src_cube, self.target_cube)
            regridded_da = xr.DataArray.from_iris(regridder(src_cube))
            regridded_var_attrs = ds[variable].attrs | {
                "grid_mapping": self.target_ds[self.target_cube.var_name].attrs[
                    "grid_mapping"
                ]
            }

            vars.update(
                {
                    variable: (
                        [
                            "time",
                            self.target_ds.cf["Y"].name,
                            self.target_ds.cf["X"].name,
                        ],
                        regridded_da.values,
                        regridded_var_attrs,
                    )
                }
            )

        # add grid mapping data from target grid
        vars.update(
            {
                target_grid_mapping: (
                    self.target_ds.cf["grid_mapping"].dims,
                    self.target_ds.cf["grid_mapping"].values,
                    self.target_ds.cf["grid_mapping"].attrs,
                )
            }
        )

        # if working with CPM data on rotated pole grid then copy the grid lat and lon bnds data too
        # if "rotated_latitude_longitude" in self.target_ds.cf.grid_mapping_names:
        vars.update(
            {
                f"{key}_bnds": (
                    [key, "bnds"],
                    self.target_ds[f"{key}_bnds"].values,
                    self.target_ds[f"{key}_bnds"].attrs,
                )
                for key in [
                    self.target_ds.cf["X"].name,
                    self.target_ds.cf["Y"].name,
                ]
            }
        )
        vars.update(
            {
                key: (
                    ["time", "bnds"],
                    ds[key].values,
                    ds[key].attrs,
                    {
                        k: ds["time"].encoding[k]
                        for k in ["units", "calendar"]
                        if k in ds["time"].encoding
                    },
                )
                for key in ["time_bnds"]
            }
        )

        coords = dict(ds.coords)
        # grid coord names are determined by the target but other coordinates should come from the source dataset
        for coord_name in ["latitude", "longitude", "grid_latitude", "grid_longitude"]:
            coords.pop(coord_name, None)
        for axis in ["X", "Y"]:
            coords[self.target_ds.cf[axis].name] = self.target_ds.cf[axis]

        ds = xr.Dataset(vars, coords=coords, attrs=ds.attrs)

        new_attrs = {"domain": "uk"}
        if (
            ds.attrs.get("resolution", "") != self.target_grid_resolution
            and ds.attrs.get("resolution", "") != ""
        ):
            new_attrs["resolution"] = (
                f"{ds.attrs.get('resolution', '')}-{self.target_grid_resolution}"
            )
        ds = ds.assign_attrs(new_attrs)

        return ds

    def _source_coord_sys(self, ds):
        """
        Determine the source coordinate system for a dataset.
        """
        if "latitude_longitude" in ds.cf.grid_mapping_names:
            src_coord_sys = iris.coord_systems.GeogCS(
                ds.cf["grid_mapping"].attrs["earth_radius"]
            )
        elif "rotated_latitude_longitude" in ds.cf.grid_mapping_names:
            src_coord_sys = iris.coord_systems.RotatedGeogCS(
                ds.cf["grid_mapping"].attrs["grid_north_pole_latitude"],
                ds.cf["grid_mapping"].attrs["grid_north_pole_longitude"],
                ellipsoid=iris.coord_systems.GeogCS(
                    ds.cf["grid_mapping"].attrs["earth_radius"]
                ),
            )
        else:
            logging.warning(
                "Unrecognised grid system. Assuming lat-lon, GeogCS(6371229.0)"
            )
            src_coord_sys = iris.coord_systems.GeogCS(6371229.0)

        return src_coord_sys

    def _da_to_iris(self, da, src_coord_sys):
        """
        Convert an xarray DataArray to an iris Cube for regridding.
        """
        src_cube = da.to_iris()

        # conversion to iris loses the coordinate system on the lat and long dimensions but iris it needs to do regrid
        src_cube.coords(axis="X")[0].coord_system = src_coord_sys
        src_cube.coords(axis="Y")[0].coord_system = src_coord_sys

        return src_cube
