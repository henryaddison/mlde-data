from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # take environment variables from .env

DATA_PATH = Path(os.getenv("DATA_PATH"))

DATASETS_PATH = DATA_PATH / "datasets"
VARIABLES_PATH = DATA_PATH / "variables"
MOOSE_VARIABLES_PATH = VARIABLES_PATH / "raw" / "moose"
DERIVED_VARIABLES_PATH = VARIABLES_PATH / "derived"


class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range):  # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)
