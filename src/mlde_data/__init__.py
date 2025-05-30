import os
from pathlib import Path


class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range):  # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)


DATA_PATH = Path(os.getenv("DATA_PATH"))

DERIVED_DATA = Path(os.getenv("DERIVED_DATA", DATA_PATH / "derived"))

MOOSE_DATA = Path(os.getenv("MOOSE_DATA", DATA_PATH / "raw" / "moose"))
