import logging
import os

import click
import xarray as xr
import yaml

from ml_downscaling_emulator import UKCPDatasetMetadata
from ml_downscaling_emulator.evaluation import load_model, predict, open_test_set

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

@click.group()
def cli():
    pass

def experiment_run_id(arch, dataset, loss="mse", epochs=200):
    return f"{arch}-{dataset}-{loss}-{epochs}-epochs"

def model_job_id(arch, dataset, loss="mse", epochs=200):
    experiments_log_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "experiments.yml")

    with open(experiments_log_filepath, 'r') as file:
        experiments = yaml.safe_load(file)

    run_id = experiment_run_id(arch, dataset, loss, epochs)
    return str(experiments[run_id]["job_id"])

def model_path(base_dir, arch, dataset, loss="mse", epochs=200):
    job_id = model_job_id(arch, dataset, loss, epochs)

    return os.path.join(base_dir, "checkpoints", arch, job_id, f"model-epoch{epochs-1}.pth")

def evalution_output_dirpath(output_base_dir, dataset, arch, portion):
    return os.path.join(output_base_dir, "evaluation", dataset, arch, portion)

@cli.command()
@click.option('--arch', type=click.STRING, required=True)
@click.option('--dataset', type=click.STRING, required=True)
@click.option('--loss', type=click.STRING, default="mse")
@click.option('--epochs', type=click.INT, default=200)
@click.argument('remote', type=click.STRING)
@click.argument('output-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def download_model(arch, dataset, loss, epochs, remote, output_base_dir):
    from fabric import Connection

    remote_host, remote_base_dir = remote.split(":")
    remote_checkpoint_path = model_path(remote_base_dir, arch, dataset, loss, epochs)

    local_checkpoint_filepath = os.path.join(output_base_dir, os.path.relpath(remote_checkpoint_path, remote_base_dir))

    logger.info(f"Downloading {remote_host}:{remote_checkpoint_path} to {local_checkpoint_filepath}")
    Connection(remote_host).get(remote_checkpoint_path, local_checkpoint_filepath)

@cli.command()
@click.option('--arch', type=click.STRING, required=True)
@click.option('--dataset', type=click.STRING, required=True)
@click.option('--loss', type=click.STRING, default="mse")
@click.option('--epochs', type=click.INT, default=200)
@click.option('--portion', type=click.STRING, default="test")
@click.argument('input-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
@click.argument('output-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def save_predictions(arch, dataset, loss, epochs, portion, input_base_dir, output_base_dir):
    dataset_dirpath = os.path.join(input_base_dir, 'nc-datasets', dataset)
    dataset_filepath = os.path.join(dataset_dirpath, f"{portion}.nc")

    ds = open_test_set(dataset_filepath)

    path = model_path(input_base_dir, arch, dataset, loss, epochs)
    model = load_model(path)

    predictions = predict(model, ds)

    output_dirpath = evalution_output_dirpath(output_base_dir, dataset, arch, portion)
    output_filepath = os.path.join(output_dirpath, "predictions.nc")

    logger.info(f"Saving predictions to {output_filepath}")
    os.makedirs(output_dirpath, exist_ok=True)
    predictions.to_netcdf(output_filepath)


@cli.command()
@click.option('--arch', type=click.STRING, required=True)
@click.option('--dataset', type=click.STRING, required=True)
@click.option('--portion', type=click.STRING, default="test")
@click.argument('input-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
@click.argument('output-base-dir', type=click.Path(exists=True), envvar='DERIVED_DATA')
def mean_diff(arch, dataset, portion, input_base_dir, output_base_dir):
    dataset_dirpath = os.path.join(input_base_dir, 'nc-datasets', dataset)
    dataset_filepath = os.path.join(dataset_dirpath, f"{portion}.nc")

    ds = open_test_set(dataset_filepath)

    eval_data_dirpath = evalution_output_dirpath(output_base_dir, dataset, arch, portion)
    predictions_filepath = os.path.join(eval_data_dirpath, "predictions.nc")

    predictions = xr.open_dataset(predictions_filepath)

    model_mean_diff = predictions.mean(dim=["time"]).pr - ds.mean(dim=["time"]).target_pr

    output_dirpath = eval_data_dirpath
    output_filepath = os.path.join(output_dirpath, "mean-diff.nc")

    logger.info(f"Saving mean diff to {output_filepath}")
    os.makedirs(output_dirpath, exist_ok=True)
    model_mean_diff.to_netcdf(output_filepath)
