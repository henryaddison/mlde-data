from collections import defaultdict
import logging
import numpy as np
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSplit(BaseSplit):
    def run(self, time_da: xr.DataArray) -> dict[str, xr.DataArray]:
        rng = np.random.default_rng(seed=self.seed)

        splits = defaultdict(list)
        for tp in self.time_periods:
            tp_time_da = time_da.sel(time=slice(tp[0], tp[1]))

            tc = np.unique(tp_time_da.dt.floor("D"))

            ntimes = len(tc)

            rng.shuffle(tc)
            split_sizes = {
                split: int(ntimes * prop)
                for split, prop in self.props.items()
                if split != "train"
            }
            split_sizes["train"] = ntimes - sum(split_sizes.values())

            for split, split_size in split_sizes.items():
                split_times = tc[:split_size]
                tc = tc[split_size:]
                splits[split].append(split_times)
            assert len(tc) == 0, "Some times were not assigned to a split"

            return {k: np.sort(np.concatenate(v)) for k, v in splits.items()}
