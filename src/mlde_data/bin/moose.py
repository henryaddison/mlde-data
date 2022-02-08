import logging
import os
from pathlib import Path
from typing import Optional

import iris
import typer

from ml_downscaling_emulator.data.moose import select_query, moose_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

app = typer.Typer()

@app.callback()
def callback():
    pass

@app.command()
def extract(variable: str, year: int, temporal_res: str = typer.Argument("daily")):
    """
    Extract data from moose
    """
    from pathlib import Path

    query = select_query(year=year, variable=variable, temporal_res=temporal_res)

    output_dirpath = Path(os.getenv("HOME"))/"data"/f"{temporal_res}_{variable}_{year}"
    query_filepath = output_dirpath/"searchfile"
    data_dirpath = output_dirpath/"data"

    os.makedirs(output_dirpath, exist_ok=True)
    os.makedirs(data_dirpath, exist_ok=True)

    typer.echo(query)
    query_filepath.write_text(query)

    moose_uri = moose_path(variable, year, ensemble_member=0, temporal_res="daily")

    query_cmd = ["moo" , "select", query_filepath, moose_uri, os.path.join(data_dirpath,"")]

    typer.echo(query_cmd)
    os.execvp(query_cmd[0], query_cmd)

@app.command()
def convert(pp_dirpath: Path, output_filepath: Path):
    """
    Convert pp data to nc
    """
    iris.save(iris.load(pp_dirpath/"*.pp"), output_filepath)
