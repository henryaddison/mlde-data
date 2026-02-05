from mlde_data.actions.regrid import Regrid
from pathlib import Path
import typer
import xarray as xr

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def main(
    ctx: typer.Context,
    filepath_source: Path,
    filepath_out: Path,
    file_target: Path = Path(
        "/home/henry/furflex/code/mlde-data/pr_rcp85_land-cpm_uk_5km_01_day_19801201-19901130.nc"
    ),
):
    """
    Regrid data using the Regrid action.

    Args:
        ctx: Typer context.
        filepath_source: Path to the source data file.
        filepath_out: Path to save the regridded output.
        file_target: Path to the target grid file.
    """
    regrid_action = Regrid(
        target_grid_resolution=file_target,
        variables=["pr"],
        scheme="nn",
    )

    # Load source dataset
    ds = xr.open_dataset(filepath_source)

    # Perform regridding
    regridded_ds = regrid_action(ds)

    # Save the regridded dataset
    regridded_ds.to_netcdf(filepath_out)


if __name__ == "__main__":
    app()


# filepath_source = Path(
#     "/home/henry/data/furflex/main/derived/moose/land-cpm/birmingham-64/2.2km-coarsened-4x/rcp85/01/pr/day/pr_rcp85_land-cpm_birmingham-64_2.2km-coarsened-4x_01_day_19801201-19811130.nc"
# )

# regrid_action = Regrid(
#     target_grid_resolution="/home/henry/furflex/code/mlde-data/src/mlde_data/actions/target_grids/2.2km/uk/moose_grid.nc",
#     variables=["pr"],
#     scheme="nn",
# )

# file_target: Path = Path(
#     "/home/henry/furflex/code/mlde-data/pr_rcp85_land-cpm_uk_5km_01_day_19801201-19901130.nc"
# )

# # Load source dataset
# ds = xr.open_dataset(filepath_source)

# # Perform regridding
# regridded_ds = regrid_action(ds)

# # Save the regridded dataset
# regridded_ds.to_netcdf("2.2km-pr.nc")
