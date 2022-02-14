VARIABLE_CODES = {
    "day": {
        "temp": {
            "stash": 30204,
            "stream": "apb"
        },
        "psl": {
            "stash": 16222,
            "stream": "apa"
        },
        "hwindu": {
            "stash": 30201,
            "stream": "apb"
        },
        "hwindv": {
            "stash": 30202,
            "stream": "apb"
        },
        "spechum": {
            "stash": 30205,
            "stream": "apb"
        },
        "1.5mtemp": {
            "stash": 3236,
            "stream": "apb" #! BAD STREAM, check! - apa based on trial and error
        },
        "pr": {
            "stash": 5216,
            "stream": "apb" #! BAD STREAM, check!
        },
        "geopotheight": {
            "stash": 30207,
            "stream": "apb" #! BAD STREAM, check!
        },
        # the saturated wet-bulb and wet-bulb potential temperatures
        "wet-bulb": {
            "stash": 16205, # 17 pressure levels for day
            "stream": "apb" #! BAD STREAM, check!
        },
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
    stream_code = VARIABLE_CODES[frequency][variable]["stream"]
    return f"moose:crum/{suite_id}/{stream_code}.pp"

def select_query(year, variable, frequency="day"):
    stash_code = VARIABLE_CODES[frequency][variable]["stash"]

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
