import logging
import numpy as np
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSplit(BaseSplit):
    def run(self, time_da: xr.DataArray) -> dict[str, xr.DataArray]:
        tc = np.unique(time_da.dt.floor("D"))
        ntimes = len(tc)

        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(tc)
        split_sizes = {
            split: int(ntimes * prop)
            for split, prop in self.props.items()
            if split != "train"
        }
        split_sizes["train"] = ntimes - sum(split_sizes.values())
        splits = {}
        for split, split_size in split_sizes.items():
            split_times = tc[:split_size]
            tc = tc[split_size:]

            splits[split] = sorted(split_times)
        assert len(tc) == 0, "Some times were not assigned to a split"
        return splits
