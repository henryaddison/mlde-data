from re import M


VARIABLE_CODES = {
    "temp": {
        "query": {
            "stash": 30204
        },
        "stream": {"day": "apb", "3hrinst": "aph",},
        "moose_name": "air_temperature"
    },
    "psl": {
        "query": {
            "stash": 16222,
        },
        "stream": {"day": "apa", "3hrinst": "apc", "6hr": "apc"},
        "moose_name": "air_pressure_at_sea_level"
    },
    "xwind": {
        "query": {
            "stash": 30201,
        },
        "stream": {"day": "apb", "3hrinst": "apg", "1hrinst": "apr"},
        "moose_name": "x_wind"
    },
    "ywind": {
        "query": {
            "stash": 30202,
        },
        "stream": {"day": "apb", "3hrinst": "apg", "1hrinst": "apr"},
        "moose_name": "y_wind"
    },
    "spechum": {
        "query": {
            "stash": 30205,
        },
        "stream": {"day": "apb", "3hrinst": "aph"},
        "moose_name": "specific_humidity"
    },
    "tmean150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 128,
        },
        "stream": {"day": "apa", "1hr": "ape"},
        "moose_name": "air_temperature"
    },
    "tmax150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 8192,
        },
        "stream": {"day": "apa"},
        "moose_name": "air_temperature"
    },
    "tmin150cm": {
        "query": {
            "stash": 3236,
            "lbproc": 4096,
        },
        "stream": {"day": "apa"},
        "moose_name": "air_temperature"
    },
    "wetbulbpott": { # the saturated wet-bulb and wet-bulb potential temperatures
        "query": {
            "stash": 16205, # 17 pressure levels for day
        },
        "stream": {"3hrinst": "aph", "1hrinst": "apr", "6hrinst": "apc"},
        "moose_name": "wet_bulb_potential_temperature"
    },
    "geopotential_height": {
        "query": {
            "stash": 30207,
        },
        "stream": {"3hrinst": "aph"}
    },
    "lsrain": {
        "query": {
            "stash": 4203,
        },
        "stream": {"day": "apa"},
        "moose_name": "stratiform_rainfall_flux"
    },
    "lssnow": {
        "query": {
            "stash": 4204,
        },
        "stream": {"day": "apa"},
        "moose_name": "stratiform_snowfall_flux"
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

TS1 = range(1980, 2001)
TS2 = range(2020, 2041)
TS3 = range(2061, 2081)

SUITE_IDS = {
    # r001i1p00000
    1: RangeDict({
        TS1: "mi-bb171",
        TS2: "mi-bb188",
        TS3: "mi-bb189",
    }),
}

# Suite ids for other ensemble members
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

def moose_path(variable, year, ensemble_member=1, frequency="day"):
    suite_id = SUITE_IDS[ensemble_member][year]
    stream_code = VARIABLE_CODES[variable]["stream"][frequency]
    return f"moose:crum/{suite_id}/{stream_code}.pp"

def select_query(year, variable, frequency="day"):
    query_conditions = VARIABLE_CODES[variable]["query"]

    def query_lines(qcond, qyear, qmonths):
        return ["begin"] + [f"    {k}={v}" for k, v in dict(yr=qyear, mon=qmonths, **qcond).items()] + ["end"]

    query_parts = [ "\n".join(query_lines(query_conditions, qyear, qmonths)) for (qyear, qmonths) in [(year-1, "12"), (year, "[1..11]")] ]

    return "\n\n".join(query_parts).lstrip()+"\n"
