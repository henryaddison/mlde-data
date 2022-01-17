import logging
from typing import Optional
import os

import typer

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

    output_dirpath = Path(os.getenv("HOME"))/f"{temporal_res}_{variable}_{year}"
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
def convert():
    """
    Convert pp data to nc
    """
    typer.echo("Converting")

variable_codes = {
    "daily": {
        "temp": {
            "stash": 30204,
            "stream": "apb"
        },
        "psl": {
            "stash": 16222,
            "stream": "apb"
        },
        "hwind_u": {
            "stash": 30201,
            "stream": "apb"
        },
        "hwind_v": {
            "stash": 30202,
            "stream": "apb"
        },
        "specific humidity": {
            "stash": 30205,
            "stream": "apb"
        },
        "1.5m temperature": {
            "stash": 3236,
            "stream": "apb"
        },
        "pr": {
            "stash": 5216,
            "stream": "apb"
        },
        "geopotential height": {
            "stash": 30207,
            "stream": "apb"
        },
        # "wet bulb": 16205 # 17 pressure levels for daily,
        # the saturated wet-bulb and wet-bulb potential temperatures
    },
}

class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range): # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)

suite_ids = {
    0: RangeDict({
        range(1980, 2001): "mi-bb171",
        range(2020, 2041): "mi-bb188",
        range(2061, 2081): "mi-bb189",
    }),
}

def moose_path(variable, year, ensemble_member=0, temporal_res="daily"):
    suite_id = suite_ids[ensemble_member][year]
    stream_code = variable_codes[temporal_res][variable]["stream"]
    return f"moose:crum/{suite_id}/{stream_code}.pp"

def select_query(year, variable, temporal_res="daily"):
    stash_code = variable_codes[temporal_res][variable]["stash"]

    return f"""
begin
    stash={stash_code}
    yr={year-1}
    mon=12
end

begin
    stash={stash_code}
    yr={year}
    mon=[1..11]
end
""".lstrip()
