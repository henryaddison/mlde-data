from enum import Enum

import typer

class DomainOption(str, Enum):
    uk = "uk"
    london = "london"

from ml_downscaling_emulator.bin import ceda
from ml_downscaling_emulator.bin import dataset
from ml_downscaling_emulator.bin import moose
from ml_downscaling_emulator.bin import preprocess

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")

if __name__ == "__main__":
    app()