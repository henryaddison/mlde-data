import logging
import os
from pathlib import Path
import shutil
import subprocess

from codetiming import Timer
import iris
import tempfile
import typer
import xarray as xr

from mlde_utils import VariableMetadata, RAW_MOOSE_VARIABLES_PATH

from ..bin.options import CollectionOption
from ..moose import (
    open_pp_data,
    select_query,
    moose_path,
    load_cubes,
    MoosePPVariableMetadata,
)

iris.FUTURE.save_split_attrs = True

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


def _domain_and_resolution_from_collection(collection: CollectionOption):
    if collection == CollectionOption.cpm:
        resolution = "2.2km"
        domain = "uk"
    elif collection == CollectionOption.gcm:
        resolution = "60km"
        domain = "global"
    else:
        raise f"Unknown collection {collection}"
    return domain, resolution


def _clean_pp_data(
    variable: str,
    year: int,
    frequency: str,
    collection: CollectionOption,
    ensemble_member: str,
    scenario: str,
    domain: str,
    resolution: str,
):
    pp_path = MoosePPVariableMetadata(
        base_dir=RAW_MOOSE_VARIABLES_PATH / "pp",
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    ).ppdata_dirpath(year)
    typer.echo(f"Removing {pp_path}...")
    shutil.rmtree(pp_path, ignore_errors=True)


def _clean_nc_data(
    variable: str,
    year: int,
    frequency: str,
    collection: CollectionOption,
    ensemble_member: str,
    scenario: str,
    domain: str,
    resolution: str,
):
    nc_path = VariableMetadata(
        base_dir=RAW_MOOSE_VARIABLES_PATH,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    ).filepath(year)
    typer.echo(f"Removing {nc_path}...")
    if os.path.exists(nc_path):
        os.remove(nc_path)


@app.command()
@Timer(name="extract", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def extract(
    collection: CollectionOption = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    year: int = typer.Option(...),
    variable: str = typer.Option(...),
    frequency: str = "day",
    base_dir: Path = None,
):
    """
    Extract data from moose
    """
    if base_dir is None:
        base_dir = RAW_MOOSE_VARIABLES_PATH / "pp"

    domain, resolution = _domain_and_resolution_from_collection(collection)

    query = select_query(
        year=year, variable=variable, frequency=frequency, collection=collection.value
    )

    moose_pp_varmeta = MoosePPVariableMetadata(
        base_dir=base_dir,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    )

    output_dirpath = moose_pp_varmeta.moose_extract_dirpath(year)

    query_filepath = Path(output_dirpath) / "searchfile"
    pp_dirpath = moose_pp_varmeta.ppdata_dirpath(year)

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
    cubes = load_cubes(str(os.path.join(pp_dirpath, "*.pp")), variable, collection)
    for cube in cubes:
        assert cube.coord("time").shape[0] == FREQ2TIMELEN[frequency]


@app.command()
@Timer(name="convert", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def convert(
    collection: CollectionOption = typer.Option(...),
    scenario: str = "rcp85",
    ensemble_member: str = typer.Option(...),
    year: int = typer.Option(...),
    variable: str = typer.Option(...),
    frequency: str = "day",
    input_base_dir: Path = None,
    output_base_dir: Path = None,
    validate: bool = True,
):
    """
    Convert pp data to a netCDF file
    """
    if input_base_dir is None:
        input_base_dir = RAW_MOOSE_VARIABLES_PATH / "pp"
    if output_base_dir is None:
        output_base_dir = RAW_MOOSE_VARIABLES_PATH

    domain, resolution = _domain_and_resolution_from_collection(collection)

    src_cubes = open_pp_data(
        base_dir=input_base_dir,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
        year=year,
    )

    output_var_meta = VariableMetadata(
        base_dir=output_base_dir,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    )
    output_filepath = output_var_meta.filepath(year)

    typer.echo(f"Saving to {output_filepath}...")
    # ensure target directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Write to a local temporary file first, then move into place. This
    # avoids HDF5/netCDF file-locking issues on NFS mounts where locks
    # are unreliable. We use a system temp dir and shutil.move so the
    # move works across filesystems.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpf:
            tmp_path = tmpf.name

        # Save to the temporary file
        iris.save(src_cubes, tmp_path)

        # Move the completed file into the final location. Use shutil.move
        # to handle cross-filesystem renames (e.g., local /tmp -> NFS).
        shutil.move(tmp_path, output_filepath)
        tmp_path = None
    finally:
        # Cleanup any leftover temp file on error
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if validate:
        assert len(xr.open_dataset(output_filepath).time) == FREQ2TIMELEN[frequency]


@app.command()
def clean(
    variable: str = typer.Option(...),
    year: int = typer.Option(...),
    frequency: str = "day",
    collection: CollectionOption = typer.Option(...),
    ensemble_member: str = typer.Option(...),
    scenario: str = "rcp85",
):
    """
    Remove any unneccessary files once processing is done
    """
    domain, resolution = _domain_and_resolution_from_collection(collection)
    _clean_pp_data(
        collection=collection,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        year=year,
        domain=domain,
        resolution=resolution,
    )
    _clean_nc_data(
        collection=collection,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        year=year,
        domain=domain,
        resolution=resolution,
    )
