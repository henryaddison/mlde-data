import typer

from ml_downscaling_emulator.bin import moose
from ml_downscaling_emulator.bin import dataset


app = typer.Typer()
app.add_typer(moose.app, name="moose")
app.add_typer(dataset.app, name="dataset")

if __name__ == "__main__":
    app()