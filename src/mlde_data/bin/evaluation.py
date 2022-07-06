import logging
import os
from pathlib import Path

import typer
import yaml

from ml_downscaling_emulator.training import load_data, get_transform
from ml_downscaling_emulator.evaluation import load_model, predict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

def model_job_id(run_id):
    experiments_log_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "experiments.yml")

    with open(experiments_log_filepath, 'r') as file:
        experiments = yaml.safe_load(file)

    return str(experiments[run_id]["job_id"])

def job_dirpath(base_dir, job_id):
    return os.path.join(base_dir, "u-net", job_id)

def model_path(base_dir, job_id, checkpoint_epoch):
    return os.path.join(job_dirpath(base_dir, job_id), "checkpoints", f"model-epoch{checkpoint_epoch}.pth")

def evalution_output_dirpath(base_dir, job_id, checkpoint_epoch, split):
    return os.path.join(job_dirpath(base_dir, job_id), "samples", f"epoch{checkpoint_epoch}", split)

app = typer.Typer()

@app.callback()
def callback():
    pass

@app.command()
def download_model(
    remote: str,
    output_base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    job_id: str = typer.Option(...),
    checkpoint: int = typer.Option(...)
):
    from fabric import Connection

    remote_host, remote_base_dir = remote.split(":")
    remote_checkpoint_path = model_path(remote_base_dir, job_id, checkpoint)

    local_checkpoint_filepath = os.path.join(output_base_dir, os.path.relpath(remote_checkpoint_path, remote_base_dir))

    logger.info(f"Downloading {remote_host}:{remote_checkpoint_path} to {local_checkpoint_filepath}")
    Connection(remote_host).get(remote_checkpoint_path, local_checkpoint_filepath)

@app.command()
def save_predictions(
    base_dir: Path = typer.Argument(..., envvar="DERIVED_DATA"),
    job_id: str = typer.Option(...),
    checkpoint: int = typer.Option(...),
    dataset: str = typer.Option(...),
    split: str = "val"
):
    dataset_dirpath = os.path.join(base_dir, 'moose', 'nc-datasets', dataset)

    _, target_transform, _ = get_transform(dataset_dirpath)
    _, eval_dl = load_data(dataset_dirpath, 64, eval_split=split)

    path = model_path(base_dir, job_id, checkpoint)
    model = load_model(path)

    predictions = predict(model, eval_dl, target_transform)

    output_dirpath = evalution_output_dirpath(base_dir, job_id, checkpoint, split)
    output_filepath = os.path.join(output_dirpath, "predictions.nc")

    logger.info(f"Saving predictions to {output_filepath}")
    os.makedirs(output_dirpath, exist_ok=True)
    predictions.to_netcdf(output_filepath)
