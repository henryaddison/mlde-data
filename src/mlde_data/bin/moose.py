import logging
import os
from pathlib import Path
import shutil
import subprocess

from codetiming import Timer
import iris
import numpy as np
import typer
import xarray as xr

from mlde_utils import VariableMetadata

from .. import MOOSE_DATA
from ..bin.options import CollectionOption
from ..moose import (
    select_query,
    moose_path,
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


class MoosePPVariableMetadata(VariableMetadata):
    """
    Extends VariableMetadata to support extra file and directory paths used when extracting pp data from Moose.
    """

    def __init__(
        self,
        variable: str,
        frequency: str,
        domain: str,
        resolution: str,
        ensemble_member: str,
        scenario: str,
        collection: str,
        base_dir: str = None,
    ):
        if base_dir is None:
            base_dir = MOOSE_DATA / "pp"
        super().__init__(
            base_dir,
            variable,
            frequency,
            domain,
            resolution,
            ensemble_member,
            scenario,
            collection,
        )

    def subdir(self):
        # collection is not included in the dirpath of parent class
        return os.path.join(self.collection, super().subdir())

    def moose_extract_dirpath(self, year):
        return os.path.join(self.dirpath(), str(year))

    def ppdata_dirpath(self, year):
        return os.path.join(self.moose_extract_dirpath(year), "data")

    def pp_files_glob(self, year):
        return os.path.join(self.ppdata_dirpath(year), "*.pp")


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
        base_dir=MOOSE_DATA,
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
):
    """
    Extract data from moose
    """
    domain, resolution = _domain_and_resolution_from_collection(collection)

    query = select_query(
        year=year, variable=variable, frequency=frequency, collection=collection.value
    )

    moose_pp_varmeta = MoosePPVariableMetadata(
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
    cube = _load_cube(str(os.path.join(pp_dirpath, "*.pp")), variable, collection)
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
):
    """
    Convert pp data to a netCDF file
    """
    domain, resolution = _domain_and_resolution_from_collection(collection)

    input_moose_pp_varmeta = MoosePPVariableMetadata(
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    )

    output_var_meta = VariableMetadata(
        base_dir=MOOSE_DATA,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    )
    output_filepath = output_var_meta.filepath(year)

    src_cube = _load_cube(
        str(input_moose_pp_varmeta.pp_files_glob(year)), variable, collection
    )

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
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    iris.save(src_cube, output_filepath)

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
