import logging
import os

import click

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.preprocessing.coarsen import Coarsen
from ml_downscaling_emulator.preprocessing.select_region import SelectRegion
from ml_downscaling_emulator.preprocessing.intensity_split import IntensitySplit
from ml_downscaling_emulator.preprocessing.random_split import RandomSplit


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

@click.group()
def cli():
    pass

@cli.command()
@click.option('--resolution', type=click.STRING, required=True)
@click.option('--domain', type=click.STRING, default="uk")
@click.option('--variable', type=click.STRING, required=True)
@click.option('--region', type=click.Choice(['london'], case_sensitive=False), required=True)
@click.option('--frequency', type=click.STRING, default='day')
@click.option('--scenario', type=click.STRING, default='rcp85')
@click.option('--ensemble-member', type=click.STRING, default='01')
@click.argument('input-base-dir', type=click.Path(exists=True))
@click.argument('output-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def select_region(input_base_dir, output_base_dir, **params):
    """Select a region within a given dataset"""

    if params["resolution"].startswith('2.2km'):
        subregion_defn = SelectRegion.LONDON_IN_CPM
    elif params["resolution"].startswith('60km'):
        subregion_defn = SelectRegion.LONDON_IN_CPM

    input_ds_params = {k: params[k] for k in ["domain", "resolution", "ensemble_member", "scenario", "variable", "frequency"]}
    input = UKCPDatasetMetadata(input_base_dir, **input_ds_params)
    output_ds_params = input_ds_params.copy()
    output_ds_params.update({"domain": params["region"]})
    output = UKCPDatasetMetadata(output_base_dir, **output_ds_params)

    click.echo(input.filepath_prefix())

    os.makedirs(output.dirpath(), exist_ok=True)

    for year in input.years():
        SelectRegion(input.filepath(year), output.filepath(year), subregion_defn).run()

@cli.command()
@click.option('--resolution', type=click.STRING, required=True)
@click.option('--domain', type=click.STRING, required=True)
@click.option('--variable', type=click.STRING, required=True)
@click.option('--scale-factor', type=click.INT, required=True)
@click.option('--frequency', type=click.STRING, default='day')
@click.option('--scenario', type=click.STRING, default='rcp85')
@click.option('--ensemble-member', type=click.STRING, default='01')
@click.argument('input-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
@click.argument('output-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def coarsen(input_base_dir, output_base_dir, **params):
    """Coarsen provided provided dataset"""
    input_ds_params = {k: params[k] for k in ["domain", "resolution", "ensemble_member", "scenario", "variable", "frequency"]}
    input = UKCPDatasetMetadata(input_base_dir, **input_ds_params)
    output_ds_params = input_ds_params.copy()
    output_ds_params.update({"resolution": f"{params['resolution']}-coarsened-{params['scale_factor']}x"})
    output = UKCPDatasetMetadata(output_base_dir, **output_ds_params)

    os.makedirs(output.dirpath(), exist_ok=True)

    for year in input.years():
        Coarsen(input.filepath(year), output.filepath(year), scale_factor = params["scale_factor"], variable = input.variable).run()

@cli.command()
@click.option('--resolution', type=click.STRING, required=True)
@click.option('--target-resolution', type=click.STRING, default="2.2km")
@click.option('--domain', type=click.STRING, required=True)
@click.option('--variable', type=click.STRING, required=True, multiple=True)
@click.option('--split-scheme', type=click.Choice(['intensity', 'random'], case_sensitive=False), required=True)
@click.option('--target-variable', type=click.STRING, default='pr')
@click.option('--val-prop', type=click.FLOAT, default=0.2)
@click.option('--test-prop', type=click.FLOAT, default=0.1)
@click.option('--frequency', type=click.STRING, default='day')
@click.option('--scenario', type=click.STRING, default='rcp85')
@click.option('--ensemble-member', type=click.STRING, default='01')
@click.argument('output-base-dir', type=click.Path(), envvar='DERIVED_DATA')
@click.argument('input-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def train_test_split(output_base_dir, input_base_dir, **params):
    """Merge"""
    conditioning_ds = []
    for variable in params["variable"]:
        conditioning_ds_params = {k: params[k] for k in ["domain", "resolution", "ensemble_member", "scenario", "frequency"]}
        conditioning_ds.append(UKCPDatasetMetadata(input_base_dir, variable=variable, **conditioning_ds_params))

    target_ds_params = {k: params[k] for k in ["domain", "ensemble_member", "scenario", "frequency"]}
    target_ds_params["resolution"] = params["target_resolution"]
    target_ds_params["variable"] = params["target_variable"]
    target_ds = UKCPDatasetMetadata(input_base_dir, **target_ds_params)

    lo_res_files = [filepath for ds_desc in conditioning_ds for filepath in ds_desc.existing_filepaths()]
    hi_res_files = target_ds.existing_filepaths()

    output_subdir = "_".join([params["resolution"], params["domain"], "-".join(params["variable"]), params["split_scheme"]])
    output_dir = os.path.join(output_base_dir, "nc-datasets", output_subdir)

    os.makedirs(output_dir, exist_ok=True)

    if params["split_scheme"] == "random":
        RandomSplit(lo_res_files, hi_res_files, output_dir, params["variable"], params["val_prop"], params["test_prop"]).run()
    elif params["split_scheme"] == "intensity":
        IntensitySplit(lo_res_files, hi_res_files, output_dir, params["variable"], params["val_prop"], params["test_prop"]).run()
    else:
        raise(f"Unknown split scheme {params['split_scheme']}")
