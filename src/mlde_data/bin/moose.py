import glob
from importlib_resources import files
import logging
import os
from pathlib import Path
import shutil
import subprocess
import yaml

import iris
import typer
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.bin import DomainOption
from ml_downscaling_emulator.data.moose import VARIABLE_CODES, select_query, moose_path
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.constrain import Constrain
from ml_downscaling_emulator.preprocessing.regrid import Regrid
from ml_downscaling_emulator.preprocessing.resample import Resample
from ml_downscaling_emulator.preprocessing.select_domain import SelectDomain
from ml_downscaling_emulator.preprocessing.sum import Sum
from ml_downscaling_emulator.preprocessing.vorticity import Vorticity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

@app.callback()
def callback():
    pass

def moose_extract_dirpath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return Path(os.getenv("MOOSE_DATA"))/"pp"/domain/resolution/"rcp85"/"01"/variable/frequency/str(year)

def ppdata_dirpath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return moose_extract_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)/"data"

def nc_filename(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return f"{variable}_rcp85_land-cpm_{domain}_{resolution}_01_{frequency}_{year-1}1201-{year}1130.nc"

def raw_nc_filepath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return Path(os.getenv("MOOSE_DATA"))/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)

def processed_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str):
    return Path(os.getenv("DERIVED_DATA"))/"moose"/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)

@app.command()
def extract(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Extract data from moose
    """
    query = select_query(year=year, variable=variable, frequency=frequency)

    output_dirpath = moose_extract_dirpath(variable=variable, year=year, frequency=frequency)
    query_filepath = output_dirpath/"searchfile"
    pp_dirpath = ppdata_dirpath(variable=variable, year=year, frequency=frequency)

    os.makedirs(output_dirpath, exist_ok=True)
    # remove any previous attempt at extracting the data (or else moo select will complain)
    shutil.rmtree(pp_dirpath, ignore_errors=True)
    os.makedirs(pp_dirpath, exist_ok=True)

    logger.debug(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(variable, year, frequency=frequency)

    query_cmd = ["moo" , "select", query_filepath, moose_uri, os.path.join(pp_dirpath,"")]

    logger.debug(f"Running {query_cmd}")
    logger.info(f"Extracting {variable} for {year}...")

    output = subprocess.run(query_cmd, capture_output=True, check=True)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    # os.execvp(query_cmd[0], query_cmd)

@app.command()
def convert(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Convert pp data to a netCDF file
    """
    pp_files_glob = ppdata_dirpath(variable=variable, year=year, frequency=frequency)/"*.pp"
    output_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency)

    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    iris.save(iris.load(str(pp_files_glob)), output_filepath)

    assert len(xr.open_dataset(output_filepath).time) == 360

@app.command()
def preprocess(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", scale_factor: int = typer.Option(...), subdomain: DomainOption = DomainOption.london, target_frequency: str = "day"):
    """
    Pre-process the moose data:
        1. Re-name the variable
        2. Re-sample the data to a match a target frequency
        3. Coarsen by given scale-factor
        4. Select a subdomain
    """
    input_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency)

    ds = xr.load_dataset(input_filepath)

    if "moose_name" in VARIABLE_CODES[variable]:
        logger.info(f"Renaming {VARIABLE_CODES[variable]['moose_name']} to {variable}...")
        ds = ds.rename({VARIABLE_CODES[variable]["moose_name"]: variable})

    if frequency != target_frequency:
        ds = Resample(target_frequency=target_frequency).run(ds)

    if scale_factor != 1:
        typer.echo(f"Coarsening {scale_factor}x...")
        target_resolution = f"2.2km-coarsened-{scale_factor}x"
        uncoarsened_ds = ds.clone()
        ds = Coarsen(scale_factor=scale_factor).run(ds)
        ds = Regrid(target_grid=uncoarsened_ds, variable=variable).run(ds)
    else:
        target_resolution = "2.2km"

    typer.echo(f"Select {subdomain.value} subdomain...")
    ds = SelectDomain(subdomain=subdomain.value).run(ds)

    assert len(ds.grid_latitude) == 64
    assert len(ds.grid_longitude) == 64

    output_filepath = processed_nc_filepath(variable=variable, year=year, frequency=target_frequency, domain=subdomain.value, resolution=target_resolution)
    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    ds.to_netcdf(output_filepath)

@app.command()
def clean(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Remove any unneccessary files once processing is done
    """
    typer.echo(f"Removing {ppdata_dirpath(variable=variable, year=year, frequency=frequency)}...")
    shutil.rmtree(ppdata_dirpath(variable=variable, year=year, frequency=frequency), ignore_errors=True)
    typer.echo(f"Removing {raw_nc_filepath(variable=variable, year=year, frequency=frequency)}...")
    os.remove(raw_nc_filepath(variable=variable, year=year, frequency=frequency))

@app.command()
def create_variable(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", domain: DomainOption = DomainOption.london, scenario="rcp85", scale_factor: int = typer.Option(...), target_resolution: str = "2.2km"):
    """
    Create a new variable from moose data
    """
    config = files('ml_downscaling_emulator.config').joinpath(f'variables/day/{variable}.yml').read_text()
    config = yaml.safe_load(config)

    sources = {}

    for source in config['sources']['moose']:

        source_nc_filepath = raw_nc_filepath(variable=source, year=year, frequency=frequency)
        logger.info(f"Opening {source_nc_filepath}")
        ds = xr.open_dataset(source_nc_filepath)

        if "moose_name" in VARIABLE_CODES[source]:
            logger.info(f"Renaming {VARIABLE_CODES[source]['moose_name']} to {source}...")
            ds = ds.rename({VARIABLE_CODES[source]["moose_name"]: source})

        sources[source] = ds

    logger.info(f"Combining {config['sources']}...")
    ds = xr.combine_by_coords(sources.values(), compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all")

    variable_resolution = "2.2km"

    for job_spec in config['spec']:
        if job_spec['action'] == "sum":
            logger.info(f"Summing {job_spec['variables']}")
            ds = Sum(job_spec['variables'], variable).run(ds)
            ds[variable] = ds[variable].assign_attrs(config['attrs'])
        elif job_spec['action'] == "coarsen":
            if scale_factor != 1:
                typer.echo(f"Coarsening {scale_factor}x...")
                variable_resolution = f"{variable_resolution}-coarsened-{scale_factor}x"
                ds, orig_ds = Coarsen(scale_factor=scale_factor).run(ds)
        elif job_spec['action'] == "regrid":
            if scale_factor != 1:
                target_grid_filepath = os.path.join(os.path.dirname(__file__), '..', 'utils', 'target-grids', target_resolution, 'uk', 'moose_pr_grid.nc')
                # orig_da = orig_ds[list(sources.keys())[0]]
                ds = Regrid(target_grid_filepath, variable=variable).run(ds)
        elif job_spec['action'] == "vorticity":
            ds = Vorticity().run(ds)
        elif job_spec['action'] == "select-subdomain":
            typer.echo(f"Select {domain.value} subdomain...")
            ds = SelectDomain(subdomain=domain.value).run(ds)
        elif job_spec['action'] == "constrain":
            ds = Constrain(query=job_spec['query']).run(ds)
        else:
            raise f"Unknown action {job_spec['action']}"
    if domain == DomainOption.london and target_resolution == "2.2km":
        assert len(ds.grid_latitude) == 64
        assert len(ds.grid_longitude) == 64

    data_basedir = os.path.join(os.getenv("DERIVED_DATA"), "moose")

    output_metadata = UKCPDatasetMetadata(data_basedir, frequency=frequency, domain=domain.value, resolution=variable_resolution, ensemble_member='01', variable=config['variable'])

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)
    ds.to_netcdf(output_metadata.filepath(year))
