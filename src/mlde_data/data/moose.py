VARIABLE_CODES = {
    "daily": {
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
            "stash": 16205, # 17 pressure levels for daily
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

SUITE_IDS = {
    0: RangeDict({
        range(1980, 2001): "mi-bb171",
        range(2020, 2041): "mi-bb188",
        range(2061, 2081): "mi-bb189",
    }),
}

def moose_path(variable, year, ensemble_member=0, temporal_res="daily"):
    suite_id = SUITE_IDS[ensemble_member][year]
    stream_code = VARIABLE_CODES[temporal_res][variable]["stream"]
    return f"moose:crum/{suite_id}/{stream_code}.pp"

def select_query(year, variable, temporal_res="daily"):
    stash_code = VARIABLE_CODES[temporal_res][variable]["stash"]

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
