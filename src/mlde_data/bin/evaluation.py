import logging
import os

import click
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

@click.group()
def cli():
    pass

@cli.command()
@click.option('--arch', type=click.STRING, required=True)
@click.option('--dataset', type=click.STRING, required=True)
@click.option('--loss', type=click.STRING, default="mse")
@click.option('--epochs', type=click.INT, default=200)
@click.argument('experiments-log-filepath', type=click.Path(exists=True))
@click.argument('remote', type=click.STRING)
def download_model(arch, dataset, loss, epochs, experiments_log_filepath, remote):
    from fabric import Connection

    experiment_run_id = f"{arch}-{dataset}-{loss}-{epochs}-epochs"
    with open(experiments_log_filepath, 'r') as file:
        experiments = yaml.safe_load(file)
    job_id = str(experiments[experiment_run_id]["job_id"])

    remote_host, remote_base_dir = remote.split(":")
    remote_checkpoint_path = os.path.join(remote_base_dir, "checkpoints", arch, job_id, f"model-epoch{epochs-1}.pth")

    local_checkpoint_dirpath = os.path.join("checkpoints", arch, job_id, '')

    logger.info(f"Downloading {remote_host}:{remote_checkpoint_path} to {local_checkpoint_dirpath}")
    Connection(remote_host).get(remote_checkpoint_path, local_checkpoint_dirpath)
