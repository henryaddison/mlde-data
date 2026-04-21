import cf_xarray  # noqa:F401
import xarray as xr
import logging
import os
import typer

app = typer.Typer()

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@app.command()
def main(
    dataset_name: str = "engwales_ccpm-4x-cpmgem_12em_future_1hr_pr",
    split: str = "test",
    shard: bool = False,
):
    logger.info(f"Rechunking {dataset_name}...")

    for var_group in ["predictors", "predictands"]:
        logger.info(f"Rechunking {split} {var_group}...")
        ds = xr.open_dataset(
            os.path.join(
                os.getenv("DATA_PATH"),
                "datasets",
                dataset_name,
                split,
                f"{var_group}.zarr",
            )
        )

        if var_group == "predictands":
            time_chunk_size = 24
        else:
            time_chunk_size = 1

        for var_name in ds.data_vars:
            new_chunks = {
                "ensemble_member": 1,
                "time": time_chunk_size,
                ds.cf["X"].name: ds.cf["X"].size,
                ds.cf["Y"].name: ds.cf["Y"].size,
            }
            if set(new_chunks.keys()) == set(ds[var_name].dims):
                del ds[var_name].encoding[
                    "chunks"
                ]  # remove existing chunking info to avoid conflicts
                ds[var_name] = ds[var_name].chunk(new_chunks)
                if shard:
                    ds[var_name].encoding["shards"] = {
                        "ensemble_member": 1,
                        "time": 100 * time_chunk_size,
                        ds.cf["X"].name: ds.cf["X"].size,
                        ds.cf["Y"].name: ds.cf["Y"].size,
                    }

        ds.to_zarr(
            os.path.join(
                os.getenv("DATA_PATH"),
                "datasets",
                f"{dataset_name}-tchunk1",
                split,
                f"{var_group}.zarr",
            ),
            mode="w-",
        )
        logger.info(f"DONE for {split} {var_group}...")


if __name__ == "__main__":
    app()
