from enum import Enum
import logging
import os
from pathlib import Path
import shutil
from typing import Optional

import iris
import typer
import xarray as xr

from ml_downscaling_emulator.data.moose import select_query, moose_path
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.select_domain import SelectDomain

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

class SubDomainOption(str, Enum):
    london = "london"

@app.callback()
def callback():
    pass

def variable_dirpath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return Path(os.getenv("MOOSE_DATA"))/domain/resolution/"rcp85"/"01"/variable/frequency/str(year)

def data_dirpath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return variable_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)/"data"

def ppdata_dirpath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return data_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)/"pp"

def nc_filename(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return f"{variable}_rcp85_land-cpm_{domain}_{resolution}_01_{frequency}_{year-1}1201-{year}1130.nc"

def raw_nc_filepath(variable: str, year: int, frequency: str, domain: str = "uk", resolution: str = "2.2km"):
    return data_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)

def processed_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str):
    return Path(os.getenv("DERIVED_DATA"))/"moose"/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution)

@app.command()
def extract(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Extract data from moose
    """
    query = select_query(year=year, variable=variable, frequency=frequency)

    output_dirpath = variable_dirpath(variable=variable, year=year, frequency=frequency)
    query_filepath = variable_dirpath(variable=variable, year=year, frequency=frequency)/"searchfile"
    pp_dirpath = ppdata_dirpath(variable=variable, year=year, frequency=frequency)

    os.makedirs(output_dirpath, exist_ok=True)
    # remove any previous attempt at extracting the data (or else moo select will complain)
    shutil.rmtree(pp_dirpath)
    os.makedirs(pp_dirpath, exist_ok=True)

    typer.echo(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(variable, year, ensemble_member=1, frequency="day")

    query_cmd = ["moo" , "select", query_filepath, moose_uri, os.path.join(ppdata_dirpath(variable=variable, year=year, frequency=frequency),"")]

    logger.debug(f"Running {query_cmd}")
    logger.info(f"Extracting {variable} for {year}...")
    os.execvp(query_cmd[0], query_cmd)

@app.command()
def convert(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Convert pp data to a netCDF file
    """
    pp_files_glob = ppdata_dirpath(variable=variable, year=year, frequency=frequency)/"*.pp"
    output_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency)

    typer.echo(f"Saving to {output_filepath}...")
    iris.save(iris.load(str(pp_files_glob)), output_filepath)

@app.command()
def preprocess(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", scale_factor: int = typer.Option(...), subdomain: SubDomainOption = SubDomainOption.london):
    """
    Coarsen data by given scale-factor
    """
    input_filepath = raw_nc_filepath(variable=variable, year=year, frequency=frequency)
    output_filepath = processed_nc_filepath(variable=variable, year=year, frequency=frequency, domain=subdomain.value, resolution=f"2.2km-coarsened-{scale_factor}x")
    ds = xr.load_dataset(input_filepath)

    if subdomain == SubDomainOption.london:
        subdomain_defn = SelectDomain.LONDON_IN_CPM_64x64

    typer.echo(f"Coarsening {scale_factor}x...")
    ds = Coarsen(scale_factor=scale_factor, variable=variable).run(ds)
    typer.echo(f"Select {subdomain.value} subdomain...")
    ds = SelectDomain(subdomain_defn=subdomain_defn).run(ds)

    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    ds.to_netcdf(output_filepath)

@app.command()
def clean(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day"):
    """
    Remove any unneccessary files once conversion is done
    """
    typer.echo(f"Removing {ppdata_dirpath(variable=variable, year=year, frequency=frequency)}...")
    shutil.rmtree(ppdata_dirpath(variable=variable, year=year, frequency=frequency))
    typer.echo(f"Removing {raw_nc_filepath(variable=variable, year=year, frequency=frequency)}...")
    os.remove(raw_nc_filepath(variable=variable, year=year, frequency=frequency))
