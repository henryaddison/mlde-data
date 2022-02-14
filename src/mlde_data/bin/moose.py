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

from enum import Enum

import typer

class SubDomainOption(str, Enum):
    london = "london"

@app.callback()
def callback():
    pass

def variable_dirpath(variable: str, year: int, temporal_res: str):
    return Path(os.getenv("MOOSE_DATA"))/f"{temporal_res}_{variable}_{year}"

def data_dirpath(variable: str, year: int, temporal_res: str):
    return variable_dirpath(variable=variable, year=year, temporal_res=temporal_res)/"data"

def ppdata_dirpath(variable: str, year: int, temporal_res: str):
    return data_dirpath(variable=variable, year=year, temporal_res=temporal_res)/"pp"

def nc_filename(variable: str, year: int, temporal_res: str, domain: str = "uk"):
    return f"{variable}_rcp85_land-cpm_{domain}_2.2km_01_{temporal_res}_{year-1}1201-{year}1130.nc"

def raw_nc_filepath(variable: str, year: int, temporal_res: str):
    return data_dirpath(variable=variable, year=year, temporal_res=temporal_res)/nc_filename(variable=variable, year=year, temporal_res=temporal_res)

def processed_nc_filepath(variable: str, year: int, temporal_res: str, domain: str = "uk"):
    return Path(os.getenv("DERIVED_DATA"))/"moose"/variable/nc_filename(variable=variable, year=year, temporal_res=temporal_res, domain=domain)

@app.command()
def extract(variable: str, year: int, temporal_res: str = typer.Argument("day")):
    """
    Extract data from moose
    """
    query = select_query(year=year, variable=variable, temporal_res=temporal_res)

    output_dirpath = Path(os.getenv("MOOSE_DATA"))/f"{temporal_res}_{variable}_{year}"
    query_filepath = variable_dirpath(variable=variable, year=year, temporal_res=temporal_res)/"searchfile"

    os.makedirs(output_dirpath, exist_ok=True)
    os.makedirs(ppdata_dirpath(variable=variable, year=year, temporal_res=temporal_res), exist_ok=True)

    typer.echo(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(variable, year, ensemble_member=1, temporal_res="day")

    query_cmd = ["moo" , "select", query_filepath, moose_uri, os.path.join(ppdata_dirpath(variable=variable, year=year, temporal_res=temporal_res),"")]

    typer.echo("Running ", query_cmd)
    os.execvp(query_cmd[0], query_cmd)

@app.command()
def convert(variable: str, year: int, temporal_res: str = typer.Argument("day")):
    """
    Convert pp data to a netCDF file
    """
    pp_files_glob = ppdata_dirpath(variable=variable, year=year, temporal_res=temporal_res)/"*.pp"
    output_filepath = raw_nc_filepath(variable=variable, year=year, temporal_res=temporal_res)

    typer.echo(f"Saving to {output_filepath}...")
    iris.save(iris.load(str(pp_files_glob)), output_filepath)

@app.command()
def preprocess(variable: str, year: int, temporal_res: str = typer.Argument("day"), scale_factor: int = typer.Option(...), subdomain: SubDomainOption = SubDomainOption.london):
    """
    Coarsen data by given scale-factor
    """
    input_filepath = raw_nc_filepath(variable=variable, year=year, temporal_res=temporal_res)
    output_filepath = processed_nc_filepath(variable=variable, year=year, temporal_res=temporal_res, domain=subdomain)
    ds = xr.load_dataset(input_filepath)

    if subdomain == SubDomainOption.london:
        subdomain_defn = SelectDomain.LONDON_IN_CPM_64x64

    typer.echo(f"Coarsening {scale_factor}x...")
    ds = Coarsen(scale_factor=scale_factor, variable=variable).run(ds)
    typer.echo(f"Select {subdomain} subdomain...")
    ds = SelectDomain(subdomain_defn=subdomain_defn).run(ds)

    typer.echo(f"Saving to {output_filepath}...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    ds.to_netcdf(output_filepath)

@app.command()
def clean(variable: str, year: int, temporal_res: str = typer.Argument("day")):
    """
    Remove any unneccessary files once conversion is done
    """
    typer.echo(f"Removing {ppdata_dirpath(variable=variable, year=year, temporal_res=temporal_res)}...")
    shutil.rmtree(ppdata_dirpath(variable=variable, year=year, temporal_res=temporal_res))
    typer.echo(f"Removing {raw_nc_filepath(variable=variable, year=year, temporal_res=temporal_res)}...")
    os.remove(raw_nc_filepath(variable=variable, year=year, temporal_res=temporal_res))
