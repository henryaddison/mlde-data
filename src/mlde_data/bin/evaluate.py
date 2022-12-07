from codetiming import Timer
import logging
import os
from pathlib import Path
from knockknock import slack_sender
import shortuuid
import torch
import typer
import xarray as xr
import yaml

import ml_downscaling_emulator.sampling as sampling
from ml_downscaling_emulator.training import restore_checkpoint
from ml_downscaling_emulator.training.dataset import get_dataset, load_raw_dataset_split
from ml_downscaling_emulator.unet import unet

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel('INFO')

app = typer.Typer()

@app.callback()
def callback():
    pass

@app.command()
# @Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
# @slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample(
        workdir: Path,
        dataset: str = typer.Option(...),
        epoch: int = typer.Option(...),
        batch_size: int = typer.Option(...),
        num_samples: int = 1,
    ):

    config_path = os.path.join(workdir, "config.yml")
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.unsafe_load(f)

    split = "val"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    output_dirpath = workdir/"samples"/f"epoch-{epoch}"/dataset/split
    os.makedirs(output_dirpath, exist_ok=True)

    transform_dir = os.path.join(workdir, "transforms")
    eval_dl, _, target_transform = get_dataset(dataset, config["dataset"], config["input_transform_key"], config["target_transform_key"], transform_dir, batch_size=batch_size, split=split, evaluation=True)

    ckpt_filename = os.path.join(workdir, "checkpoints", f"epoch_{epoch}.pth")
    num_predictors, _, _ = eval_dl.dataset[0][0].shape
    model = unet.UNet(num_predictors, 1).to(device=device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters())
    state = dict(step=0, epoch=0, optimizer=optimizer, model=model)
    state = restore_checkpoint(ckpt_filename, state, device)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        xr_samples = sampling.sample(state["model"], eval_dl, target_transform)

        output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

        logger.info(f"Saving predictions to {output_filepath}")
        os.makedirs(output_dirpath, exist_ok=True)
        xr_samples.to_netcdf(output_filepath)

@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample_id(
        workdir: Path,
        dataset: str = typer.Option(...),
        split:str = "val",
        num_samples: int = 1,
    ):

    output_dirpath = workdir/"samples"/"epoch-0"/dataset/split
    os.makedirs(output_dirpath, exist_ok=True)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        eval_ds = load_raw_dataset_split(dataset, split)
        samples = eval_ds["pr"].values
        predictions = sampling.np_samples_to_xr(samples, eval_ds, target_transform=None)

        output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

        logger.info(f"Saving predictions to {output_filepath}")
        os.makedirs(output_dirpath, exist_ok=True)
        predictions.to_netcdf(output_filepath)
