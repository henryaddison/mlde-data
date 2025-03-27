import logging
import os
import shutil
import subprocess

from codetiming import Timer
import iris
import numpy as np
import typer
import xarray as xr

from ..bin.options import CollectionOption
from ..moose import (
    select_query,
    moose_path,
    moose_extract_dirpath,
    moose_cache_dirpath,
    ppdata_dirpath,
    raw_nc_filepath,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s: %(message)s")

app = typer.Typer()


@app.callback()
def callback():
    pass


FREQ2TIMELEN = {
    "day": 360,
    "1hr": 360 * 24,
}


def _load_cube(pp_files, variable, collection):
    if variable == "pr" and collection == CollectionOption.gcm:
        # for some reason precip extract for GCM has a mean and max hourly cell method version
        # only want the mean version
        constraint = iris.Constraint(
            cube_func=lambda cube: cube.cell_methods[0].method == "mean"
        )
    else:
        constraint = None
    return iris.load_cube(pp_files, constraint=constraint)


@app.command()
@Timer(name="extract", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def extract(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = "day",
    collection: CollectionOption = typer.Option(...),
    ensemble_member: str = typer.Option(...),
    cache: bool = True,
):
    """
    Extract data from moose
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    cache_path = moose_cache_dirpath(
        variable=variable,
        year=year,
        frequency=frequency,
        collection=collection.value,
        resolution=resolution,
        domain=domain,
        ensemble_member=ensemble_member,
    )
    cache_check_filepath = cache_path / ".cache-ready"

    if cache:
        if os.path.exists(cache_check_filepath):
            logger.info(f"Moose cache available {cache_path}")
            return

    query = select_query(
        year=year, variable=variable, frequency=frequency, collection=collection.value
    )

    output_dirpath = moose_extract_dirpath(
        variable=variable,
        year=year,
        frequency=frequency,
        resolution=resolution,
        collection=collection.value,
        domain=domain,
        ensemble_member=ensemble_member,
        cache=False,
    )
    query_filepath = output_dirpath / "searchfile"
    pp_dirpath = ppdata_dirpath(
        variable=variable,
        year=year,
        frequency=frequency,
        resolution=resolution,
        collection=collection.value,
        domain=domain,
        ensemble_member=ensemble_member,
        cache=False,
    )

    os.makedirs(output_dirpath, exist_ok=True)
    # remove any previous attempt at extracting the data (or else moo select will complain)
    shutil.rmtree(pp_dirpath, ignore_errors=True)
    os.makedirs(pp_dirpath, exist_ok=True)

    logger.debug(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(
        variable,
        year,
        frequency=frequency,
        collection=collection.value,
        ensemble_member=ensemble_member,
    )

    query_cmd = [
        "moo",
        "select",
        query_filepath,
        moose_uri,
        os.path.join(pp_dirpath, ""),
    ]

    logger.debug(f"Running {query_cmd}")
    logger.info(f"Extracting {variable} for {year}...")

    output = subprocess.run(query_cmd, capture_output=True, check=False)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    output.check_returncode()

    # make sure have the correct amount of data from moose
    cube = _load_cube(str(pp_dirpath / "*.pp"), variable, collection)
    assert cube.coord("time").shape[0] == FREQ2TIMELEN[frequency]

    if cache:
        cache_path = moose_cache_dirpath(
            variable=variable,
            year=year,
            frequency=frequency,
            collection=collection.value,
            resolution=resolution,
            domain=domain,
            ensemble_member=ensemble_member,
        )
        logger.info(f"Copying {output_dirpath} to {cache_path}...")
        os.makedirs(cache_path, exist_ok=True)
        shutil.copytree(output_dirpath, cache_path, dirs_exist_ok=True)
        cache_check_filepath.touch()


@app.command()
@Timer(name="convert", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def convert(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = "day",
    collection: CollectionOption = typer.Option(...),
    ensemble_member: str = typer.Option(...),
    cache: bool = True,
):
    """
    Convert pp data to a netCDF file
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    cache_path = moose_cache_dirpath(
        variable=variable,
        year=year,
        frequency=frequency,
        collection=collection.value,
        resolution=resolution,
        domain=domain,
        ensemble_member=ensemble_member,
    )
    cache_check_filepath = cache_path / ".cache-ready"

    if cache:
        if os.path.exists(cache_check_filepath):
            logger.info(f"Moose cache available {cache_path}")
        else:
            logger.info(f"Moose cache unavailable {cache_path}")
            cache = False

    pp_files_glob = (
        ppdata_dirpath(
            variable=variable,
            year=year,
            frequency=frequency,
            resolution=resolution,
            collection=collection.value,
            domain=domain,
            cache=cache,
            ensemble_member=ensemble_member,
        )
        / "*.pp"
    )
    output_filepath = raw_nc_filepath(
        variable=variable,
        year=year,
        frequency=frequency,
        resolution=resolution,
        collection=collection.value,
        domain=domain,
        ensemble_member=ensemble_member,
    )

    src_cube = _load_cube(str(pp_files_glob), variable, collection)

    # bug in some data means the final grid_latitude bound is very large (1.0737418e+09)
    if collection == CollectionOption.cpm and any(
        [variable.startswith(var) for var in ["xwind", "ywind", "spechum", "temp"]]
    ):
        bounds = np.copy(src_cube.coord("grid_latitude").bounds)
        # make sure it really is much larger than expected (in case this gets fixed)
        assert bounds[-1][1] > 8.97
        bounds[-1][1] = 8.962849
        src_cube.coord("grid_latitude").bounds = bounds

    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    iris.save(src_cube, output_filepath)

    assert len(xr.open_dataset(output_filepath).time) == FREQ2TIMELEN[frequency]


@app.command()
def clean(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = "day",
    collection: CollectionOption = typer.Option(...),
    ensemble_member: str = typer.Option(...),
):
    """
    Remove any unneccessary files once processing is done
    """
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"

    pp_path = ppdata_dirpath(
        variable=variable,
        year=year,
        frequency=frequency,
        collection=collection.value,
        resolution=resolution,
        domain=domain,
        ensemble_member=ensemble_member,
        cache=False,
    )
    typer.echo(f"Removing {pp_path}...")
    shutil.rmtree(pp_path, ignore_errors=True)
    raw_nc_path = raw_nc_filepath(
        variable=variable,
        year=year,
        frequency=frequency,
        collection=collection.value,
        resolution=resolution,
        domain=domain,
        ensemble_member=ensemble_member,
    )
    typer.echo(f"Removing {raw_nc_path}...")
    if os.path.exists(raw_nc_path):
        os.remove(raw_nc_path)
