import xarray as xr

for res in ["2.2km", "2.2km-coarsened-2x", "2.2km-coarsened-4x", "2.2km-coarsened-8x", "2.2km-coarsened-27x", "60km"]:
    if res == "60km":
        domain = "global"
    else:
        domain = "uk"
    path = f"src/ml_downscaling_emulator/utils/target-grids/{res}/{domain}/moose_grid.nc"
    ds = xr.load_dataset(path)
    for v in ds.variables:
        print(v, ds[v].attrs.pop("coordinates", None))
    ds.to_netcdf(path)
