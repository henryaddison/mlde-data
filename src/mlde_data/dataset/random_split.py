import logging
import numpy as np
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSplit(BaseSplit):
    def run(self, time_da: xr.DataArray) -> dict[str, xr.DataArray]:
        tc = time_da.values.copy()
        ntimes = len(tc)

        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(tc)

        splits = {}
        for split, split_prop in self.props.items():
            split_size = int(ntimes * split_prop)

            test_times = tc[:split_size]
            tc = tc[split_size:]

            split_set = time_da.where(
                time_da.time.isin(test_times) == True, drop=True  # noqa: E712
            )
            splits[split] = split_set

        return splits
