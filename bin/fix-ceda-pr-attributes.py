import dotenv
import glob
import logging
import typer
import xarray as xr

dotenv.load_dotenv()

from mlde_utils import DERIVED_VARIABLES_PATH  # noqa:E402


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main():
    pr_filepaths = glob.glob(
        str(
            DERIVED_VARIABLES_PATH
            / "land-cpm"
            / "engwales"
            / "2.2km-coarsened-4x"
            / "rcp85"
            / "*"
            / "pr"
            / "1hr"
            / "*.nc"
        )
    )
    logger.info(f"Found {len(pr_filepaths)} pr files to fix attributes for.")
    for filepath in pr_filepaths:
        logger.info(f"Fixing attributes for {filepath}...")
        ds = xr.load_dataset(filepath)
        ds["pr"].attrs["units"] = "mm/hour"
        for k in ["contact, institution", "institution_id", "references"]:
            if k in ds.attrs:
                del ds.attrs[k]
        ds = ds.drop_vars(["month_number", "year", "yyyymmddhh"])
        ds.to_netcdf(filepath, mode="w")
    logger.info("Done fixing attributes for all files.")


if __name__ == "__main__":
    app()
