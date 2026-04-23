import logging
import typer
import xarray as xr

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main():
    for split in ["train", "val", "test"]:
        ds = xr.open_dataset(
            f"/gws/ssde/j25a/furflex/henrya/projects/cpmgem-daily/data/main/derived/moose/nc-datasets/bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr/{split}.nc"
        )["time"].drop_vars("season")
        ds["time"] = ds["time"].dt.floor("D")
        ds.to_netcdf(
            f"/gws/ssde/j25a/furflex/henrya/projects/furflex/data/cpmgem-hourly/preset-dataset-splits/bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr/{split}.nc"
        )


if __name__ == "__main__":
    app()
