import typer

from ml_downscaling_emulator.bin import moose

app = typer.Typer()
app.add_typer(moose.app, name="moose")

if __name__ == "__main__":
    app()