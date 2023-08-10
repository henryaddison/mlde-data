import iris
import iris.analysis
import xarray as xr

"""
Regrid a dataset based on a given target grid file
"""


class Regrid:

    SCHEMES = {
        "linear": iris.analysis.Linear,
        "nn": iris.analysis.Nearest,
        "area-weighted": iris.analysis.AreaWeighted,
    }

    def __init__(self, target_grid_filepath, variables, scheme="nn") -> None:
        self.target_cube = iris.load_cube(target_grid_filepath)
        self.target_ds = xr.open_dataset(target_grid_filepath)
        self.variables = variables
        self.scheme = self.SCHEMES[scheme]()

    def run(self, ds):
        # regrid the coarsened data to match the original horizontal grid (using NN interpolation)
        # NB iris and xarray can only comminicate in dataarrays not datasets
        # and form a dataset based on the original hi-res with this new coarsened then NN-gridded data

        if "latitude_longitude" in ds.variables:
            src_coord_sys = iris.coord_systems.GeogCS(
                ds["latitude_longitude"].attrs["earth_radius"]
            )
            src_lat_name = "latitude"
            src_lon_name = "longitude"
        elif "rotated_latitude_longitude" in ds.variables:
            src_coord_sys = iris.coord_systems.RotatedGeogCS(
                ds["rotated_latitude_longitude"].attrs["grid_north_pole_latitude"],
                ds["rotated_latitude_longitude"].attrs["grid_north_pole_longitude"],
                ellipsoid=iris.coord_systems.GeogCS(
                    ds["rotated_latitude_longitude"].attrs["earth_radius"]
                ),
            )
            src_lat_name = "grid_latitude"
            src_lon_name = "grid_longitude"
        else:
            raise RuntimeError("Unrecognised grid system")

        if "latitude_longitude" in self.target_ds.variables:
            target_grid_mapping = "latitude_longitude"
            target_lat_name = "latitude"
            target_lon_name = "longitude"
        elif "rotated_latitude_longitude" in self.target_ds.variables:
            target_grid_mapping = "rotated_latitude_longitude"
            target_lat_name = "grid_latitude"
            target_lon_name = "grid_longitude"
        else:
            raise RuntimeError("Unrecognised grid system")

        vars = {}

        for variable in self.variables:
            src_cube = ds[variable].to_iris()
            # conversion to iris loses the coordinate system on the lat and long dimensions but iris it needs to do regrid
            src_cube.coords(src_lon_name)[0].coord_system = src_coord_sys
            src_cube.coords(src_lat_name)[0].coord_system = src_coord_sys

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
                        ["time", target_lat_name, target_lon_name],
                        regridded_da.values,
                        regridded_var_attrs,
                    )
                }
            )

        # add grid mapping data from target grid
        vars.update(
            {
                target_grid_mapping: (
                    self.target_ds[target_grid_mapping].dims,
                    self.target_ds[target_grid_mapping].values,
                    self.target_ds[target_grid_mapping].attrs,
                )
            }
        )

        # if working with CPM data on rotated pole grid then copy the grid lat and lon bnds data too
        if "rotated_latitude_longitude" in self.target_ds.variables:
            vars.update(
                {
                    f"{key}_bnds": (
                        [key, "bnds"],
                        self.target_ds[f"{key}_bnds"].values,
                        self.target_ds[f"{key}_bnds"].attrs,
                    )
                    for key in [target_lat_name, target_lon_name]
                }
            )
        vars.update(
            {
                key: (
                    ["time", "bnds"],
                    ds[key].values,
                    ds[key].attrs,
                    {"units": "hours since 1970-01-01 00:00:00", "calendar": "360_day"},
                )
                for key in ["time_bnds"]
            }
        )

        coords = dict(ds.coords)
        # grid coord names are determined by the target but other coordinates should come from the source dataset
        for coord_name in ["latitude", "longitude", "grid_latitude", "grid_longitude"]:
            coords.pop(coord_name, None)
        coords[target_lon_name] = self.target_ds.coords[target_lon_name]
        coords[target_lat_name] = self.target_ds.coords[target_lat_name]

        ds = xr.Dataset(vars, coords=coords, attrs=ds.attrs)

        return ds
