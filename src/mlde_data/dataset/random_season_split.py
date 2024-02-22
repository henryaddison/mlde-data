from collections import defaultdict
import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class RandomSeasonSplit:
    def __init__(self, time_encoding, val_prop=0.2, test_prop=0.1, seed=42) -> None:
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.time_encoding = time_encoding
        self.seed = seed

    def run(self, combined_dataset):

        split_chunks = defaultdict(list)
        for years in map(lambda x: list(range(x, x + 20)), [1981, 2021, 2061]):
            for season in ["DJF", "MAM", "JJA", "SON"]:
                rng = np.random.default_rng(seed=self.seed)
                rng.shuffle(years)

                test_year_count = int(len(years) * self.test_prop)
                val_year_count = int(len(years) * self.val_prop)
                seasonal_year_split = {
                    "test": years[:test_year_count],
                    "val": years[test_year_count : test_year_count + val_year_count],
                    "train": years[test_year_count + val_year_count :],
                }

                for split, split_years in seasonal_year_split.items():
                    if season == "DJF":
                        # for winter need to take December from previous year and Jan and Feb from other years
                        split_chunk = combined_dataset.sel(
                            time=self._inwinter(combined_dataset, split_years)
                        )
                    else:
                        # other seasons don't cross year boundary
                        split_chunk = combined_dataset.sel(
                            time=self._inseason(combined_dataset, split_years, season)
                        )

                    split_chunks[split].append(split_chunk)

        splits = {
            split: xr.merge(
                split_chunks,
                compat="no_conflicts",
                combine_attrs="no_conflicts",
            ).sortby("time")
            for split, split_chunks in split_chunks.items()
        }
        # for split_ds in splits.values():
        #     # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
        #     split_ds.time.encoding.update(self.time_encoding)

        return splits

    def _inwinter(self, ds, years):
        return ((ds["time.year"].isin(years)) & (ds["time.month"].isin([1, 2]))) | (
            (ds["time.year"].isin(list(map(lambda year: year - 1, years))))
            & (ds["time.month"] == 12)
        )

    def _inseason(self, ds, years, season):
        return (ds["time.year"].isin(years)) & (ds["time.season"] == season)
