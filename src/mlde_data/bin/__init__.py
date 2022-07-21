from enum import Enum

import typer

class DomainOption(str, Enum):
    uk = "uk"
    london = "london"
    birmingham = "birmingham"

class CollectionOption(str, Enum):
    gcm = "land-gcm"
    cpm = "land-cpm"

from . import ceda
from . import dataset
from . import evaluation
from . import moose
from . import preprocess

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(evaluation.app, name="evaluate")
app.add_typer(moose.app, name="moose")
app.add_typer(preprocess.app, name="preprocess")

if __name__ == "__main__":
    app()