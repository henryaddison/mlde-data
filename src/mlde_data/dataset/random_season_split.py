import cftime
from collections import defaultdict
import logging
import numpy as np
from typing import List
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSeasonSplit(BaseSplit):
    def __init__(
        self,
        props: dict[str, float],
        time_periods: List[List[int]],
        seed: int = 42,
    ):
        super().__init__(props=props, seed=seed)
        self.time_periods = time_periods

    def run(self, time_da: xr.DataArray) -> dict[str, List[cftime.Datetime360Day]]:

        splits = defaultdict(lambda: xr.CFTimeIndex([]))
        for season in ["DJF", "MAM", "JJA", "SON"]:
            for tp in self.time_periods:
                years = list(range(tp[0], tp[1] + 1))
                nyears = len(years)

                rng = np.random.default_rng(seed=self.seed)
                rng.shuffle(years)

                for split, split_prop in self.props.items():
                    split_year_count = int(nyears * split_prop)
                    if split_year_count > 0:
                        split_years = years[:split_year_count]
                        years = years[split_year_count:]

                        for year in split_years:
                            season_range = xr.date_range(
                                self._season_start_date(season, year),
                                periods=90,
                                freq="D",
                                use_cftime=True,
                            )
                            splits[split] = splits[split].append(season_range)

        # return {k: sorted(v) for k, v in splits.items()}
        return {
            k: xr.CFTimeIndex(v.sort_values()).intersection(
                time_da["time"].dt.floor("D")
            )
            for k, v in splits.items()
        }

    def _season_start_date(self, season: str, year: int) -> cftime.Datetime360Day:
        if season == "DJF":
            return cftime.Datetime360Day(
                year - 1, 12, 1, 0, 0, 0, 0, has_year_zero=True
            )
        elif season == "MAM":
            return cftime.Datetime360Day(year, 3, 1, 0, 0, 0, 0, has_year_zero=True)
        elif season == "JJA":
            return cftime.Datetime360Day(year, 6, 1, 0, 0, 0, 0, has_year_zero=True)
        elif season == "SON":
            return cftime.Datetime360Day(year, 9, 1, 0, 0, 0, 0, has_year_zero=True)
        else:
            raise ValueError(f"Unknown season: {season}")
