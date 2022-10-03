import logging
import os
from pathlib import Path
import subprocess
import yaml

from codetiming import Timer
import typer
import xarray as xr

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.bin import DomainOption, CollectionOption
from ml_downscaling_emulator.data.moose import VARIABLE_CODES, raw_nc_filepath, remove_forecast
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.constrain import Constrain
from ml_downscaling_emulator.preprocessing.regrid import Regrid
from ml_downscaling_emulator.preprocessing.remapcon import Remapcon
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

@app.command()
@Timer(name="create-variable", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def create_variable(config_path: Path = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", domain: DomainOption = DomainOption.london, scenario="rcp85", scale_factor: str = typer.Option(...), target_resolution: str = "2.2km", target_size: int = 64):
    """
    Create a new variable from moose data
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # add cli parameters to config
    config["parameters"] = {
        "frequency": frequency,
        "domain": domain.value,
        "scenario": scenario,
        "scale_factor": scale_factor,
        "target_resolution": target_resolution
    }

    collection = CollectionOption(config['sources']['collection'])
    if collection == CollectionOption.cpm:
        variable_resolution = "2.2km"
        source_domain = "uk"
    elif collection == CollectionOption.gcm:
        variable_resolution = "60km"
        source_domain = "global"
    else:
        raise f"Unknown collection {collection}"

    sources = {}

    # ds = xr.open_mfdataset([raw_nc_filepath(variable=source, year=year, frequency=frequency) for source in config['sources']['moose']])
    # for source in config['sources']['moose']:
    #     if "moose_name" in VARIABLE_CODES[source]:
    #         logger.info(f"Renaming {VARIABLE_CODES[source]['moose_name']} to {source}...")
    #         ds = ds.rename({VARIABLE_CODES[source]["moose_name"]: source})

    # currently only support data from moose
    assert(config["sources"]["type"] == "moose")

    for src_variable in config['sources']['variables']:
        source_nc_filepath = raw_nc_filepath(variable=src_variable["name"], year=year, frequency=src_variable["frequency"], resolution=variable_resolution, collection=collection.value, domain=source_domain)
        logger.info(f"Opening {source_nc_filepath}")
        ds = xr.open_dataset(source_nc_filepath)

        if "moose_name" in VARIABLE_CODES[src_variable["name"]]:
            logger.info(f"Renaming {VARIABLE_CODES[src_variable['name']]['moose_name']} to {src_variable['name']}...")
            ds = ds.rename({VARIABLE_CODES[src_variable["name"]]["moose_name"]: src_variable["name"]})

        # remove forecast related coords that we don't need
        ds = remove_forecast(ds)

        sources[src_variable["name"]] = ds

    logger.info(f"Combining {config['sources']}...")
    ds = xr.combine_by_coords(sources.values(), compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all")

    for job_spec in config['spec']:
        if job_spec['action'] == "sum":
            logger.info(f"Summing {job_spec['variables']}")
            ds = Sum(job_spec['variables'], config['variable']).run(ds)
            ds[config['variable']] = ds[config['variable']].assign_attrs(config['attrs'])
        elif job_spec['action'] == "coarsen":
            if scale_factor == "gcm":
                typer.echo(f"Remapping conservatively to gcm grid...")
                variable_resolution = f"{variable_resolution}-coarsened-gcm"
                # pick the target grid based on the job spec
                # some variables use one grid, others a slightly offset one
                grid_type = job_spec["parameters"]["grid"]
                target_grid_filepath = os.path.join(os.path.dirname(__file__), '..', 'utils', 'target-grids', '60km', 'global', grid_type, 'moose_grid.nc')
                ds = Remapcon(target_grid_filepath).run(ds)
            else:
                scale_factor = int(scale_factor)
                if scale_factor == 1:
                    typer.echo(f"{scale_factor}x coarsening scale factor, nothing to do...")
                else:
                    typer.echo(f"Coarsening {scale_factor}x...")
                    variable_resolution = f"{variable_resolution}-coarsened-{scale_factor}x"
                    ds, orig_ds = Coarsen(scale_factor=scale_factor).run(ds)
        elif job_spec['action'] == "regrid_to_target":
            if target_resolution != variable_resolution:
                typer.echo(f"Regridding to target resolution...")
                target_grid_filepath = os.path.join(os.path.dirname(__file__), '..', 'utils', 'target-grids', target_resolution, 'uk', 'moose_grid.nc')
                kwargs = job_spec.get("parameters", {})
                ds = Regrid(target_grid_filepath, variables=[config['variable']], **kwargs).run(ds)
        elif job_spec['action'] == "vorticity":
            typer.echo(f"Computing vorticity...")
            ds = Vorticity().run(ds)
        elif job_spec['action'] == "select-subdomain":
            typer.echo(f"Select {domain.value} subdomain...")
            ds = SelectDomain(subdomain=domain.value, size=target_size).run(ds)
        elif job_spec['action'] == "constrain":
            typer.echo(f"Filtering...")
            ds = Constrain(query=job_spec['query']).run(ds)
        elif job_spec['action'] == "rename":
            typer.echo(f"Renaming...")
            ds = ds.rename(job_spec["mapping"])
        else:
            raise f"Unknown action {job_spec['action']}"

    assert len(ds.grid_latitude) == target_size
    assert len(ds.grid_longitude) == target_size

    # there should be no missing values in this dataset
    assert ds[config["variable"]].isnull().sum().values.item() == 0

    data_basedir = os.path.join(os.getenv("DERIVED_DATA"), "moose")

    output_metadata = UKCPDatasetMetadata(data_basedir, frequency=frequency, domain=f"{domain.value}-{target_size}", resolution=f"{variable_resolution}-{target_resolution}", ensemble_member='01', variable=config['variable'])

    logger.info(f"Saving data to {output_metadata.filepath(year)}")
    os.makedirs(output_metadata.dirpath(), exist_ok=True)
    ds.to_netcdf(output_metadata.filepath(year))
    with open(os.path.join(output_metadata.dirpath(), f"{config['variable']}-{year}.yml"), 'w') as f:
        yaml.dump(config, f)

def run_cmd(cmd):
    logger.debug(f"Running {cmd}")
    output = subprocess.run(cmd, capture_output=True, check=False)
    stdout = output.stdout.decode("utf8")
    print(stdout)
    print(output.stderr.decode("utf8"))
    output.check_returncode()

@app.command()
def xfer(variable: str = typer.Option(...), year: int = typer.Option(...), frequency: str = "day", domain: DomainOption = DomainOption.london, collection: CollectionOption = typer.Option(...), resolution: str = typer.Option(...), target_size: int = 64):
    # TODO re-write xfer in Python
    jasmin_filepath = processed_nc_filepath(variable=variable, year=year, frequency=frequency, domain=f"{domain.value}-{target_size}", resolution=resolution, collection=collection.value)
    bp_filepath = processed_nc_filepath(variable=variable, year=year, frequency=frequency, domain=f"{domain.value}-{target_size}", resolution=resolution, collection=collection.value, base_dir="/user/work/vf20964")

    file_xfer_cmd = [f"{os.getenv('HOME')}/code/ml-downscaling-emulation/moose-etl/xfer-script-direct", jasmin_filepath, bp_filepath]
    config_xfer_cmd = []
    run_cmd(file_xfer_cmd)
