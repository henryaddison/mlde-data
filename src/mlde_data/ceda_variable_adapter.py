import logging
from pathlib import Path
import xarray as xr

from mlde_data.variable import SourceVariableConfig


class CedaVariableAdapter:
    """
    Adapter for opening variables from CEDA on JASMIN
    """

    RIP_CODES2CEDA_EM = {
        "r001i1p00000": "01",
        "r001i1p01113": "04",
        "r001i1p01554": "05",
        "r001i1p01649": "06",
        "r001i1p01843": "07",
        "r001i1p01935": "08",
        "r001i1p02868": "09",
        "r001i1p02123": "10",
        "r001i1p02242": "11",
        "r001i1p02305": "12",
        "r001i1p02335": "13",
        "r001i1p02491": "15",
    }

    VERSIONS = {
        "land-cpm": "v20210615",
    }

    @classmethod
    def from_variable_defn(
        cls, defn: SourceVariableConfig, ensemble_member: str, scenario: str, year: int
    ):
        if defn.src_type != "ceda":
            raise ValueError(
                f"Cannot create CedaVariableAdapter from variable definition with source type {defn.src_type}"
            )

        return cls(
            collection=defn.collection,
            variable=defn.variable,
            frequency=defn.frequency,
            resolution=defn.resolution,
            domain=defn.domain,
            ensemble_member=ensemble_member,
            scenario=scenario,
            year=year,
        )

    def __init__(
        self,
        collection: str,
        ensemble_member: str,
        variable: str,
        frequency: str,
        resolution: str,
        domain: str,
        scenario: str,
        year: int,
        base_dir: Path | None = None,
    ):
        self.collection = collection
        self.ensemble_member = ensemble_member
        self.variable = variable
        self.frequency = frequency
        self.resolution = resolution
        self.domain = domain
        self.scenario = scenario
        self.year = year
        if base_dir is None:
            base_dir = Path("/badc/ukcp18/data")
        self.base_dir = base_dir

    def __eq__(self, other):
        if not isinstance(other, CedaVariableAdapter):
            return False

        return (
            self.collection == other.collection
            and self.ensemble_member == other.ensemble_member
            and self.variable == other.variable
            and self.frequency == other.frequency
            and self.resolution == other.resolution
            and self.domain == other.domain
            and self.scenario == other.scenario
            and self.year == other.year
        )

    @property
    def _dirpath(self) -> Path:
        return (
            self.base_dir
            / self.collection
            / self.domain
            / self.resolution
            / self.scenario
            / self.RIP_CODES2CEDA_EM[self.ensemble_member]
            / self.variable
            / self.frequency
            / self.VERSIONS[self.collection]
        )

    @property
    def _filename_prefix(self) -> str:
        return "_".join(
            [
                self.variable,
                self.scenario,
                self.collection,
                self.domain,
                self.resolution,
                self.RIP_CODES2CEDA_EM[self.ensemble_member],
                self.frequency,
            ]
        )

    @property
    def _filenames(self) -> list[str]:
        # hourly data is split into monthly files, but daily data is in a single file for the whole year
        if self.frequency == "1hr":
            return [
                f"{self._filename_prefix}_{self.year-1}1201-{self.year-1}1230.nc"
            ] + [
                f"{self._filename_prefix}_{self.year}{month:02d}01-{self.year}{month:02d}30.nc"
                for month in range(1, 12)
            ]
        elif self.frequency == "day":
            return [f"{self._filename_prefix}_{self.year}1201-{self.year}1130.nc"]
        else:
            raise ValueError(f"Unknown frequency {self.frequency}")

    @property
    def filepaths(self) -> list[Path]:
        return [self._dirpath / filename for filename in self._filenames]

    def open(self) -> xr.Dataset:
        logging.debug(f"Opening {self.filepaths}")
        ds = xr.concat(
            [
                xr.open_dataset(f).sel(
                    time_counter=slice(f"{self.year-1}-12-01", f"{self.year}-11-30")
                )
                for f in self.filepaths
            ],
            dim="time_counter",
            data_vars="minimal",
            coords="minimal",
            join="exact",
            combine_attrs="no_conflicts",
            compat="identical",
        )

        return ds
