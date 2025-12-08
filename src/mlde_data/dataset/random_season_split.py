from collections import defaultdict
import logging

import numpy as np
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSeasonSplit(BaseSplit):
    def run(self, time_da: xr.DataArray) -> dict[str, xr.DataArray]:

        split_chunks = defaultdict(list)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            for years in map(lambda x: list(range(x, x + 20)), [1981, 2021, 2061]):
                nyears = len(years)
                rng = np.random.default_rng(seed=self.seed)
                rng.shuffle(years)
                for split, split_prop in self.props.items():
                    split_year_count = int(nyears * split_prop)
                    if split_year_count > 0:
                        split_years = years[:split_year_count]
                        years = years[split_year_count:]

                        split_chunk = time_da.sel(
                            time=self._inseason(time_da, split_years, season)
                        )

                        split_chunks[split].append(split_chunk)

        splits = {
            split: xr.concat(
                split_chunks,
                dim="time",
                compat="no_conflicts",
                combine_attrs="no_conflicts",
            ).sortby("time")
            for split, split_chunks in split_chunks.items()
        }

        return splits

    def _inwinter(self, ds, years):
        return ((ds["time.year"].isin(years)) & (ds["time.month"].isin([1, 2]))) | (
            (ds["time.year"].isin(list(map(lambda year: year - 1, years))))
            & (ds["time.month"] == 12)
        )

    def _inseason(self, ds, years, season):
        if season == "DJF":
            # for winter need to take December from previous year and Jan and Feb from other years
            return self._inwinter(ds, years)
        else:
            # other seasons don't cross year boundary
            return (ds["time.year"].isin(years)) & (ds["time.season"] == season)
