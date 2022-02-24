import typer


from ml_downscaling_emulator.bin import ceda
from ml_downscaling_emulator.bin import dataset
from ml_downscaling_emulator.bin import moose

app = typer.Typer()
app.add_typer(ceda.app, name="ceda")
app.add_typer(dataset.app, name="dataset")
app.add_typer(moose.app, name="moose")


if __name__ == "__main__":
    app()