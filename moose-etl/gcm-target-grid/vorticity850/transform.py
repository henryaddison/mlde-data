import os
from pathlib import Path

import iris
import xarray as xr

working_grid_pp_path = "pp-data/*.pp"
working_grid_nc_path = "moose_grid.nc"

target_grid_path = Path(
    "../../../src/ml_downscaling_emulator/utils/target-grids/60km/global/vorticity850/moose_grid.nc"
)

# convert pp data to netcdf and open with xr
target_cube = iris.load_cube(
    working_grid_pp_path, constraint=iris.Constraint(pressure=850.0)
)
iris.save(target_cube, working_grid_nc_path)
ds = xr.load_dataset(working_grid_nc_path).isel(time=0)

# remove forecast related attributes
ds = ds.reset_coords(
    ["forecast_period", "forecast_reference_time", "realization"], drop=True
).drop_vars(["forecast_period_bnds"])
for v in ds.variables:
    print(v, ds[v].encoding.pop("coordinates", None))

# save into the codebase
os.makedirs(target_grid_path.parent, exist_ok=True)
ds.to_netcdf(target_grid_path)
