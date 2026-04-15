from iris.time import PartialDateTime
import iris
from ncdata.iris_xarray import cubes_to_xarray
import numpy as np
from pathlib import Path
import xarray as xr

from mlde_data.moose import SUITE_IDS, load_cubes
from mlde_data.variable import SourceVariableConfig
from mlde_data.options import CollectionOption


class MooseVariableAdapter:
    """
    Adapter for opening variables extracted from MASS on JASMIN
    """

    MASS_EXTRACTS_BASE_DIR = Path(
        "/gws/ssde/j25a/furflex/henrya/projects/furflex/data/mass-extracts"
    )

    @classmethod
    def from_variable_defn(
        cls,
        defn: SourceVariableConfig,
        ensemble_member: str,
        scenario: str,
        year: int,
        base_dir: Path | None = None,
    ):
        if defn.src_type != "moose":
            raise ValueError(
                f"Cannot create MooseVariableAdapter from variable definition with source type {defn.src_type}"
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
            base_dir=base_dir,
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
            base_dir = self.MASS_EXTRACTS_BASE_DIR
        self.base_dir = base_dir

    def __eq__(self, other):
        if not isinstance(other, MooseVariableAdapter):
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
        suite_id = SUITE_IDS[self.collection][self.ensemble_member][self.year]
        return self.base_dir / suite_id / self.variable / "data"

    @property
    def _filenames(self) -> list[str]:
        return [f"*{self.year-1}12*.pp", f"*{self.year}*.pp"]

    @property
    def _filepaths(self) -> list[Path]:
        return [self._dirpath / fn for fn in self._filenames]

    def open(self) -> xr.Dataset:
        pdt1 = PartialDateTime(year=self.year - 1, month=12, day=1)
        pdt2 = PartialDateTime(year=self.year, month=12, day=1)
        year_constraint = iris.Constraint(time=lambda cell: pdt1 <= cell.point < pdt2)
        # realize the data (or something odd happens when saving to netcdf below)
        src_cubes = load_cubes(
            [str(fp) for fp in self._filepaths],
            self.variable,
            self.collection,
            realize=True,
            constraints=[year_constraint],
        )

        # bug in some data means the final grid_latitude bound is very large (1.0737418e+09)
        for src_cube in src_cubes:
            if self.collection == CollectionOption.cpm:
                bounds = np.copy(src_cube.coord("grid_latitude").bounds)
                # make sure it really is much larger than expected (in case this gets fixed)
                if bounds[-1][1] > 8.97:
                    bounds[-1][1] = 8.962849
                    src_cube.coord("grid_latitude").bounds = bounds

        ds = cubes_to_xarray(src_cubes)
        # for some reason cubes_to_xarray output is missing indexes on the coords
        # this used to be avoided as saving cubes to netcdf and then re-opening with xarray didn't have this problem
        # TODO: work out why much more memory is required by cubes_to_xarray compared to iris.save and xarray.open_dataset approach
        for d in ds.dims:
            ds[d] = ds[d].reindex()

        return ds
