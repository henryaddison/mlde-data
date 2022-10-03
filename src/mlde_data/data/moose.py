import os
from pathlib import Path

### CPM details

# moose uris: moose:crum/{suite_id}/{stream_code}.pp

### GCM details

# moose uris: moose:ens/{suite_id}/{rip_code}/{stream_code}.pp

# stream codes
# Data streams for 6-hourly data is apc.pp. For daily mean data it is ape.pp.

VARIABLE_CODES = {
    "temp": {
        "query": {
            "stash": 30204
        },
        "stream": {"land-cpm": {"day": "apb", "3hrinst": "aph",}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature"
    },
    "psl": {
        "query": {
            "stash": 16222,
        },
        "stream": {"land-cpm": {"day": "apa", "3hrinst": "apc", "6hr": "apc"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_pressure_at_sea_level"
    },
    "xwind": {
        "query": {
            "stash": 30201,
        },
        "stream": {"land-cpm": {"day": "apb", "3hrinst": "apg", "1hrinst": "apr"}, "land-gcm": {"day": "ape"}},
        "moose_name": "x_wind"
    },
    "ywind": {
        "query": {
            "stash": 30202,
        },
        "stream": {"land-cpm": {"day": "apb", "3hrinst": "apg", "1hrinst": "apr"}, "land-gcm": {"day": "ape"}},
        "moose_name": "y_wind"
    },
    "spechum": {
        "query": {
            "stash": 30205,
        },
        "stream": {"land-cpm": {"day": "apb", "3hrinst": "aph"}, "land-gcm": {"day": "ape"}},
        "moose_name": "specific_humidity"
    },
    "tmean150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 128,
        },
        "stream": {"land-cpm": {"day": "apa", "1hr": "ape"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature"
    },
    "tmax150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 8192,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature"
    },
    "tmin150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 4096,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "air_temperature"
    },
    "wetbulbpott": { # the saturated wet-bulb and wet-bulb potential temperatures
        "query": {
            "stash": 16205, # 17 pressure levels for day
        },
        "stream": {"land-cpm": {"3hrinst": "aph", "1hrinst": "apr", "6hrinst": "apc"}, "land-gcm": {"day": "ape"}},
        "moose_name": "wet_bulb_potential_temperature"
    },
    "geopotential_height": {
        "query": {
            "stash": 30207,
        },
        "stream": {"land-cpm": {"3hrinst": "aph"}, "land-gcm": {"day": "ape"}}
    },
    "lsrain": {
        "query": {
            "stash": 4203,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "stratiform_rainfall_flux"
    },
    "lssnow": {
        "query": {
            "stash": 4204,
        },
        "stream": {"land-cpm": {"day": "apa"}, "land-gcm": {"day": "ape"}},
        "moose_name": "stratiform_snowfall_flux"
    },
    "pr": {
        "query": {
            "stash": 5216,
        },
        "stream": {"land-gcm": {"day": "apa"}},
        "moose_name": "precipitation_flux"
    }
}

for theta in [250, 500, 700, 850, 925]:
    VARIABLE_CODES[f"spechum{theta}"] = {
        "query": {
            "stash": 30205,
            "lblev": theta,
        },
        "stream": {"land-cpm": {"day": "apb", "3hrinst": "aph"}, "land-gcm": {"day": "ape"}},
        "moose_name": "specific_humidity"
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

TS1 = range(1980, 2001)
TS2 = range(2020, 2041)
TS3 = range(2061, 2081)
TSRecent = range(1971, 2006)
TSNearF = range(2006, 2077)
TSFarF = range(2077, 2100)

SUITE_IDS = {
    "land-cpm": {
        # r001i1p00000
        1: RangeDict({
            TS1: "mi-bb171",
            TS2: "mi-bb188",
            TS3: "mi-bb189",
        }),
    },
    "land-gcm": {
        # r001i1p00000
        1: RangeDict({
            TSRecent: "u-ap977",
            TSNearF: "u-ar095",
            TSFarF: "u-au084",
        }),
    }
}

# Suite ids for other CPM ensemble members
# r001i1p01113 - {TS1: "mi-bb190", TS2: "mi-bb191", TS3: "mi-bb192"}
# r001i1p01554 - {TS1: "mi-bb193", TS2: "mi-bb194", TS3: "mi-bb195"}
# r001i1p01649 - {TS1: "mi-bb196", TS2: "mi-bb197", TS3: "mi-bb198"}
# r001i1p01843 - {TS1: "mi-bb199", TS2: "mi-bb200", TS3: "mi-bb201"}
# r001i1p01935 - {TS1: "mi-bb202", TS2: "mi-bb203", TS3: "mi-bb204"}
# r001i1p02868 - {TS1: "mi-bb205", TS2: "mi-bb206", TS3: "mi-bb208"}
# r001i1p02123 - {TS1: "mi-bb209", TS2: "mi-bb210", TS3: "mi-bb211"}
# r001i1p02242 - {TS1: "mi-bb214", TS2: "mi-bb215", TS3: "mi-bb216"}
# r001i1p02305 - {TS1: "mi-bb217", TS2: "mi-bb218", TS3: "mi-bb219"}
# r001i1p02335 - {TS1: "mi-bb220", TS2: "mi-bb221", TS3: "mi-bb222"}
# r001i1p02491 - {TS1: "mi-bb223", TS2: "mi-bb224", TS3: "mi-bb225"}

# Suite names for GCM time periods
# The four suite names covering the historical and RCP8.5 experiments are:
# Historical: u-an398 (Dec 1896 - Nov 1970), u-ap977 (Dec 1970 - Nov 2005)
# RCP8.5    : u-ar095 (Dec 2005 – Nov 2076), u-au084 (Dec 2076 – Nov 2099)

RIP_CODES = {
    "land-gcm": {
        1: "r001i1p00000"
    }
}

# other GCM rip codes
# r001i1p00000 (standard physics model) r001i1p01113  r001i1p01935  r001i1p02305 r001i1p02832
# r001i1p00090 r001i1p01554  r001i1p02089  r001i1p02335  r001i1p02868
# r001i1p00605  r001i1p01649  r001i1p02123  r001i1p02491  r001i1p02884
# r001i1p00834  r001i1p01843  r001i1p02242  r001i1p02753  r001i1p02914

def moose_path(variable, year, ensemble_member=1, frequency="day", collection="land-cpm"):
    if collection == "land-cpm":
        suite_id = SUITE_IDS[collection][ensemble_member][year]
        stream_code = VARIABLE_CODES[variable]["stream"][collection][frequency]
        return f"moose:crum/{suite_id}/{stream_code}.pp"
    elif collection == "land-gcm":
        suite_id = SUITE_IDS[collection][ensemble_member][year]
        stream_code = VARIABLE_CODES[variable]["stream"][collection][frequency]
        rip_code = RIP_CODES[collection][ensemble_member]
        return f"moose:ens/{suite_id}/{rip_code}/{stream_code}.pp"
    else:
        raise f"Unknown collection {collection}"

def select_query(year, variable, frequency="day", collection="land-cpm"):
    query_conditions = VARIABLE_CODES[variable]["query"]

    def query_lines(qcond, qyear, qmonths):
        return ["begin"] + [f"    {k}={v}" for k, v in dict(yr=qyear, mon=qmonths, **qcond).items()] + ["end"]

    query_parts = [ "\n".join(query_lines(query_conditions, qyear, qmonths)) for (qyear, qmonths) in [(year-1, "12"), (year, "[1..11]")] ]

    return "\n\n".join(query_parts).lstrip()+"\n"


def moose_extract_dirpath(variable: str, year: int, frequency: str, resolution: str, collection: str, domain: str, cache: bool = False):
    if cache:
        path_start = os.getenv("MOOSE_CACHE")
    else:
        path_start = os.getenv("MOOSE_DATA")
    return Path(path_start)/"pp"/collection/domain/resolution/"rcp85"/"01"/variable/frequency/str(year)

def moose_cache_dirpath(**kwargs):
    return moose_extract_dirpath(**kwargs, cache=True)

def ppdata_dirpath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str, cache: bool = False):
    return moose_extract_dirpath(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection, cache=cache)/"data"

def nc_filename(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str):
    return f"{variable}_rcp85_{collection}_{domain}_{resolution}_01_{frequency}_{year-1}1201-{year}1130.nc"

def raw_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str = "land-cpm"):
    return Path(os.getenv("MOOSE_DATA"))/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection)

def processed_nc_filepath(variable: str, year: int, frequency: str, domain: str, resolution: str, collection: str, base_dir=os.getenv("DERIVED_DATA")):
    return Path(base_dir)/"moose"/domain/resolution/"rcp85"/"01"/variable/frequency/nc_filename(variable=variable, year=year, frequency=frequency, domain=domain, resolution=resolution, collection=collection)

def remove_forecast(ds):
    coords_to_remove = []
    for v in ds.variables:
        if v in ["forecast_period", "forecast_reference_time", "realization"]:
            coords_to_remove.append(v)
    ds = ds.reset_coords(coords_to_remove, drop=True)

    if "forecast_period_bnds" in ds.variables:
        ds = ds.drop_vars("forecast_period_bnds", errors='ignore')

    for v in ds.variables:
        if "coordinates" in ds[v].encoding:
            new_coords_encoding = re.sub("(realization|forecast_period|forecast_reference_time) ?", "", ds[v].encoding["coordinates"]).strip()
            ds[v].encoding.update({"coordinates": new_coords_encoding})

    return ds
