from click import Path
import iris
from mlde_utils import VariableMetadata
import numpy as np
import os
import re

from . import RangeDict
from .bin.options import CollectionOption


class MoosePPVariableMetadata(VariableMetadata):
    """
    Extends VariableMetadata to support extra file and directory paths used when extracting pp data from Moose.
    """

    def __init__(
        self,
        variable: str,
        frequency: str,
        domain: str,
        resolution: str,
        ensemble_member: str,
        scenario: str,
        collection: str,
        base_dir: str,
    ):
        super().__init__(
            base_dir,
            variable,
            frequency,
            domain,
            resolution,
            ensemble_member,
            scenario,
            collection,
        )

    def moose_extract_dirpath(self, year):
        return os.path.join(self.dirpath(), str(year))

    def ppdata_dirpath(self, year):
        return os.path.join(self.moose_extract_dirpath(year), "data")

    def pp_files_glob(self, year):
        return os.path.join(self.ppdata_dirpath(year), "*.pp")


###############
# CPM details #
###############
# moose uris: moose:crum/{suite_id}/{stream_code}.pp

###############
# GCM details #
###############
# moose uris: moose:ens/{suite_id}/{rip_code}/{stream_code}.pp

# stream codes
# Data streams for 6-hourly data is apc.pp. For daily mean data it is ape.pp.

VARIABLE_CODES = {
    "psl": {
        "query": {
            "stash": 16222,
        },
        "stream": {
            "land-cpm": {"day": "apa", "3hrinst": "apc", "6hr": "apc"},
            "land-gcm": {"day": "apd"},
        },
        "moose_name": "air_pressure_at_sea_level",
    },
    "relhum150cm": {
        "query": {
            "stash": 3245,
        },
        "stream": {"land-cpm": {"day": "apa"}},
        "moose_name": "relative_humidity",
    },
    "windmax10m": {
        "query": {
            "stash": 3227,
        },
        "stream": {
            "land-cpm": {"day": "apa"},
        },
    },
    "windmean10m": {
        "query": {
            "stash": 3227,
        },
        "stream": {
            "land-cpm": {"day": "apk"},
        },
    },
    "uwind10m": {
        "query": {
            "stash": 3225,
        },
        "stream": {
            "land-cpm": {"day": "apa", "1hrinst": "apd"},
        },
    },
    "vwind10m": {
        "query": {
            "stash": 3226,
        },
        "stream": {
            "land-cpm": {"day": "apa", "1hrinst": "apd"},
        },
    },
    "tmean150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 128,
        },
        "stream": {
            "land-cpm": {"day": "apa", "1hr": "ape"},
            "land-gcm": {"day": "ape"},
        },
        "moose_name": "air_temperature",
    },
    "tmax150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 8192,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature",
    },
    "tmin150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 4096,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature",
    },
    "wetbulbpott": {  # the saturated wet-bulb and wet-bulb potential temperatures
        "query": {
            "stash": 16205,  # 17 pressure levels for day
        },
        "stream": {
            "land-cpm": {"3hrinst": "aph", "1hrinst": "apr", "6hrinst": "apc"},
            "land-gcm": {"day": "ape"},
        },
        "moose_name": "wet_bulb_potential_temperature",
    },
    "geopotential_height": {
        "query": {
            "stash": 30207,
        },
        "stream": {"land-cpm": {"3hrinst": "aph"}, "land-gcm": {"day": "ape"}},
    },
    "lsrainsnow": {
        "query": {
            "stash": "(4203, 4204)",
        },
        "stream": {
            "land-cpm": {"day": "apa", "1hr": "apq"},
            "land-gcm": {"day": "ape"},
        },
    },
    "pr": {
        "query": {
            "stash": 5216,
        },
        "stream": {"land-gcm": {"day": "apa"}},
        "moose_name": "precipitation_flux",
    },
    # a hack for some wind components for Sam
    "x_wind": {
        "query": {
            "stash": 30201,
            "lblev": "(500,850)",
        },
        "stream": {
            "land-gcm": {"day": "ape"},
        },
    },
    "y_wind": {
        "query": {
            "stash": 30202,
            "lblev": "(500,850)",
        },
        "stream": {
            "land-gcm": {"day": "ape"},
        },
    },
    "wind": {
        "query": {
            "stash": "(30201,30202)",
            "lblev": "(500,850)",
        },
        "stream": {
            "land-gcm": {"day": "ape"},
        },
    },
    "surft": {
        "query": {
            "stash": "(24,507)",
        },
        "stream": {
            "land-gcm": {"day": "apa"},
        },
    },
}

THETAS = [250, 500, 700, 850]
VARIABLE_CODES["mlqtw"] = {
    "query": {
        # 30201 - x_wind, 30202 - y_wind, 30204 - temperature, 30205 - specific humidity
        "stash": "(30201, 30202, 30204, 30205)",
        "lblev": f"({', '.join([str(t) for t in THETAS])})",
    },
    "stream": {
        "land-cpm": {"day": "apb"},
        "land-gcm": {"day": "ape"},
    },
}
# stream codes for other frequencies
# temp: land-cpm: "3hrinst": "aph",
# spechum: land-cpm: "3hrinst": "aph",
# wind: land-cpm: "3hrinst": "apg", "1hrinst": "apr"


TS1 = range(1980, 2001)
TS2 = range(2020, 2041)
TS3 = range(2061, 2081)
TSHistorical = range(1897, 1971)
TSRecent = range(1971, 2006)
TSNearF = range(2006, 2077)
TSFarF = range(2077, 2100)

SUITE_IDS = {
    "land-cpm": {
        "r001i1p00000": RangeDict({TS1: "mi-bb171", TS2: "mi-bb188", TS3: "mi-bb189"}),
        "r001i1p01113": RangeDict({TS1: "mi-bb190", TS2: "mi-bb191", TS3: "mi-bb192"}),
        "r001i1p01554": RangeDict({TS1: "mi-bb193", TS2: "mi-bb194", TS3: "mi-bb195"}),
        "r001i1p01649": RangeDict({TS1: "mi-bb196", TS2: "mi-bb197", TS3: "mi-bb198"}),
        "r001i1p01843": RangeDict({TS1: "mi-bb199", TS2: "mi-bb200", TS3: "mi-bb201"}),
        "r001i1p01935": RangeDict({TS1: "mi-bb202", TS2: "mi-bb203", TS3: "mi-bb204"}),
        "r001i1p02868": RangeDict({TS1: "mi-bb205", TS2: "mi-bb206", TS3: "mi-bb208"}),
        "r001i1p02123": RangeDict({TS1: "mi-bb209", TS2: "mi-bb210", TS3: "mi-bb211"}),
        "r001i1p02242": RangeDict({TS1: "mi-bb214", TS2: "mi-bb215", TS3: "mi-bb216"}),
        "r001i1p02305": RangeDict({TS1: "mi-bb217", TS2: "mi-bb218", TS3: "mi-bb219"}),
        "r001i1p02335": RangeDict({TS1: "mi-bb220", TS2: "mi-bb221", TS3: "mi-bb222"}),
        "r001i1p02491": RangeDict({TS1: "mi-bb223", TS2: "mi-bb224", TS3: "mi-bb225"}),
    },
    # Suite names for GCM time periods
    # The four suite names covering the historical and RCP8.5 experiments are:
    # Historical: u-an398 (Dec 1896 - Nov 1970), u-ap977 (Dec 1970 - Nov 2005)
    # RCP8.5    : u-ar095 (Dec 2005 – Nov 2076), u-au084 (Dec 2076 – Nov 2099)
    "land-gcm": RangeDict(
        {
            TSHistorical: "u-an398",
            TSRecent: "u-ap977",
            TSNearF: "u-ar095",
            TSFarF: "u-au084",
        }
    ),
}

GCM_RIP_CODES = [
    "r001i1p00000",
    "r001i1p01113",
    "r001i1p01554",
    "r001i1p01649",
    "r001i1p01843",
    "r001i1p01935",
    "r001i1p02868",
    "r001i1p02123",
    "r001i1p02242",
    "r001i1p02305",
    "r001i1p02335",
    "r001i1p02491",
    "r001i1p02832",
    "r001i1p00090",
    "r001i1p02089",
    "r001i1p00605",
    "r001i1p02884",
    "r001i1p00834",
    "r001i1p02753",
    "r001i1p02914",
]

# mapping between short-hand CPM ensemble member identifier and rip codes
# for different GCM ensemble members (also allow use of the GCM rip codes directly)
RIP_CODES = {
    "land-gcm": {
        "01": "r001i1p00000",
        "04": "r001i1p01113",
        "05": "r001i1p01554",
        "06": "r001i1p01649",
        "07": "r001i1p01843",
        "08": "r001i1p01935",
        "09": "r001i1p02868",
        "10": "r001i1p02123",
        "11": "r001i1p02242",
        "12": "r001i1p02305",
        "13": "r001i1p02335",
        "15": "r001i1p02491",
    }
    | {rc: rc for rc in GCM_RIP_CODES}
}


def moose_path(variable, year, ensemble_member, frequency="day", collection="land-cpm"):
    if collection == "land-cpm":
        suite_id = SUITE_IDS[collection][ensemble_member][year]
        stream_code = VARIABLE_CODES[variable]["stream"][collection][frequency]
        return f"moose:crum/{suite_id}/{stream_code}.pp"  # noqa: E231
    elif collection == "land-gcm":
        suite_id = SUITE_IDS[collection][year]
        stream_code = VARIABLE_CODES[variable]["stream"][collection][frequency]
        rip_code = ensemble_member
        return f"moose:ens/{suite_id}/{rip_code}/{stream_code}.pp"  # noqa: E231
    else:
        raise f"Unknown collection {collection}"


def select_query(year, variable, frequency="day", collection="land-cpm"):
    query_conditions = VARIABLE_CODES[variable]["query"]

    def query_lines(qcond, qyear, qmonths):
        return (
            ["begin"]
            + [f"    {k}={v}" for k, v in dict(yr=qyear, mon=qmonths, **qcond).items()]
            + ["end"]
        )

    query_parts = [
        "\n".join(query_lines(query_conditions, qyear, qmonths))
        for (qyear, qmonths) in [(year - 1, "12"), (year, "[1..11]")]
    ]

    return "\n\n".join(query_parts).lstrip() + "\n"


def remove_forecast(ds):
    coords_to_remove = []
    for v in ds.variables:
        if v in ["forecast_period", "forecast_reference_time", "realization"]:
            coords_to_remove.append(v)
    ds = ds.reset_coords(coords_to_remove, drop=True)

    if "forecast_period_bnds" in ds.variables:
        ds = ds.drop_vars("forecast_period_bnds", errors="ignore")

    for v in ds.variables:
        if "coordinates" in ds[v].encoding:
            new_coords_encoding = re.sub(
                "(realization|forecast_period|forecast_reference_time) ?",
                "",
                ds[v].encoding["coordinates"],
            ).strip()
            ds[v].encoding.update({"coordinates": new_coords_encoding})

    return ds


def remove_pressure(ds):
    if "pressure" in ds.variables:
        ds = ds.reset_coords("pressure", drop=True)

    for v in ds.variables:
        if "coordinates" in ds[v].encoding:
            new_coords_encoding = re.sub(
                "(pressure) ?", "", ds[v].encoding["coordinates"]
            ).strip()
            ds[v].encoding.update({"coordinates": new_coords_encoding})
    return ds


def load_cubes(pp_files, variable, collection, realize=False):
    if variable == "pr" and collection == CollectionOption.gcm:
        # for some reason precip extract for GCM has a mean and max hourly cell method version
        # only want the mean version
        constraint = iris.Constraint(
            cube_func=lambda cube: cube.cell_methods[0].method == "mean"
        )
    else:
        constraint = None

    cubes = iris.load(pp_files, constraints=constraint)

    if realize:
        for cube in cubes:
            cube.data

    return cubes


def open_pp_data(
    base_dir: Path,
    collection: CollectionOption,
    scenario: str,
    ensemble_member: str,
    variable: str,
    frequency: str,
    resolution: str,
    domain: str,
    year: int,
):
    input_moose_pp_varmeta = MoosePPVariableMetadata(
        base_dir=base_dir,
        collection=collection.value,
        scenario=scenario,
        ensemble_member=ensemble_member,
        variable=variable,
        frequency=frequency,
        resolution=resolution,
        domain=domain,
    )

    # realize the data (or something odd happens when saving to netcdf below)
    src_cubes = load_cubes(
        str(input_moose_pp_varmeta.pp_files_glob(year)),
        variable,
        collection,
        realize=True,
    )

    # bug in some data means the final grid_latitude bound is very large (1.0737418e+09)
    for src_cube in src_cubes:
        if collection == CollectionOption.cpm:
            bounds = np.copy(src_cube.coord("grid_latitude").bounds)
            # make sure it really is much larger than expected (in case this gets fixed)
            if bounds[-1][1] > 8.97:
                bounds[-1][1] = 8.962849
                src_cube.coord("grid_latitude").bounds = bounds
