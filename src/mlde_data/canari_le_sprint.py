import logging
import os
import xarray as xr
from . import RangeDict


class CanariLESprintVariable:
    CANARI_LE_BASE_PATH = os.getenv(
        "CANARI_LE_BASE_PATH", "/gws/nopw/j04/canari/shared/large-ensemble"
    )

    HIST2 = range(1950, 2015)
    SSP370 = range(2015, 2100)
    DIRS = RangeDict({HIST2: "HIST2", SSP370: "SSP370"})
    ENSEMBLE_MEMBERS = RangeDict(
        {
            HIST2: {
                "1": "cv575a",
                "22": "da190a",
                "35": "cz649a",
                "10": "cw343a",
                "23": "da191a",
                "36": "dd436a",
                "11": "cy375a",
                "24": "da192a",
                "37": "dd438a",
                "12": "cy376a",
                "25": "da193a",
                "38": "dd439a",
                "13": "cy537a",
                "26": "db291a",
                "39": "dd441a",
                "14": "cy811a",
                "27": "db301a",
                "4": "cw356a",
                "15": "cy866a",
                "28": "db303a",
                "40": "dd442a",
                "16": "cy873a",
                "29": "db304a",
                "5": "cv827a",
                "17": "cy877a",
                "3": "cw345a",
                "6": "cv976a",
                "18": "cy879a",
                "30": "db305a",
                "7": "cz547a",
                "19": "cy880a",
                "31": "cz475a",
                "8": "cy436a",
                "2": "cv625a",
                "32": "cz568a",
                "9": "cw342a",
                "20": "cy881a",
                "33": "cz647a",
                "21": "da179a",
                "34": "cz648a",
            },
            SSP370: {
                "1": "de814a",
                "22": "df934a",
                "35": "di515a",
                "10": "de851a",
                "23": "df935a",
                "36": "di703a",
                "11": "de934a",
                "24": "df936a",
                "37": "di704a",
                "12": "de937a",
                "25": "df937a",
                "38": "di705a",
                "13": "de938a",
                "26": "dh412a",
                "39": "di706a",
                "14": "de939a",
                "27": "dh413a",
                "4": "de815a",
                "15": "de940a",
                "28": "dh415a",
                "40": "di707a",
                "16": "df299a",
                "29": "dh416a",
                "5": "df220a",
                "17": "df300a",
                "3": "de724a",
                "6": "de830a",
                "18": "df301a",
                "30": "dh417a",
                "7": "de831a",
                "19": "df302a",
                "31": "di511a",
                "8": "de832a",
                "2": "de436a",
                "32": "di512a",
                "9": "de850a",
                "20": "df303a",
                "33": "di513a",
                "21": "df933a",
                "34": "di514a",
            },
        }
    )

    VARIABLES = {
        "psl": {"day": "m01s16i222_4"},
    }
    for theta in [250, 500, 700, 850, 925]:
        VARIABLES[f"xwind{theta}"] = {"day": "m01s30i201_3"}
        VARIABLES[f"ywind{theta}"] = {"day": "m01s30i201_3"}
        VARIABLES[f"temp{theta}"] = {"day": "m01s30i204_3"}

    def __init__(self, variable: str, ensemble_member: str, frequency: str, year: int):
        self.variable = variable
        self.ensemble_member = ensemble_member
        self.frequency = frequency
        self.year = year

    @property
    def varcode(self) -> str:
        return self.VARIABLES[self.variable][self.frequency]

    @property
    def filepaths(self) -> list[str]:
        # canonical year for this project runs 12-01 to 11-30
        # but for CANARI, years are split at 01-01, so need to consider
        # two files for each canonical project year
        return [
            self._filepath(canari_year) for canari_year in [self.year - 1, self.year]
        ]

    def _filepath(self, canari_year: int) -> str:
        return os.path.join(
            self.CANARI_LE_BASE_PATH,
            "priority",
            self.DIRS[self.year],
            self.ensemble_member,
            "ATM",
            "yearly",
            str(canari_year),
            self.filename(canari_year),
        )

    def _ensemble_code(self, canari_year: int) -> str:
        return self.ENSEMBLE_MEMBERS[canari_year][self.ensemble_member]

    def _filename(self, canari_year) -> str:
        return f"{self.ensemble_code(canari_year)}_{self.ensemble_member}_{self.frequency}_{self.varcode}.nc"

    def open(self) -> xr.Dataset:
        logging.info(f"Opening {self.filepaths}")
        ds = xr.combine_by_coords([xr.open_dataset(f) for f in self.filepaths])

        ds = ds.rename(
            {
                self.varcode: self.variable,
                "time_counter": "time",
                "axis_nbounds": "bnds",
            }
        )
        if "lat_um_atmos_grid_t" in ds.dims:
            ds = ds.rename(
                {
                    "lat_um_atmos_grid_t": "latitude",
                    "lon_um_atmos_grid_t": "longitude",
                    "bounds_lat_um_atmos_grid_t": "latitude_bnds",
                    "bounds_lon_um_atmos_grid_t": "longitude_bnds",
                }
            )
        if "lat_um_atmos_grid_uv" in ds.dims:
            ds = ds.rename(
                {
                    "lat_um_atmos_grid_uv": "latitude",
                    "lon_um_atmos_grid_uv": "longitude",
                    "bounds_lat_um_atmos_grid_uv": "latitude_bnds",
                    "bounds_lon_um_atmos_grid_uv": "longitude_bnds",
                }
            )

        ds = ds.assign(
            latitude_longitude=xr.DataArray(
                data=0, dims=[], coords=dict(), attrs=dict(earth_radius=6371229.0)
            )
        )
        ds[self.variable] = ds[self.variable].assign_attrs(
            grid_mapping="latitude_longitude"
        )

        ds = ds.sel(time=slice(f"{self.year-1}-12-01", f"{self.year}-12-01"))

        return ds
