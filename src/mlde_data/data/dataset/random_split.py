import logging
import os

import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
import xarray as xr

logger = logging.getLogger(__name__)

class RandomSplit:
    def __init__(self, time_encoding, val_prop=0.2, test_prop=0.1) -> None:
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.time_encoding = time_encoding

    def run(self, combined_dataset):
        tc = combined_dataset.time.values.copy()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(tc)

        test_size = int(len(tc)*self.test_prop)
        val_size = int(len(tc)*self.val_prop)

        test_times = tc[0:test_size]
        val_times = tc[test_size:test_size+val_size]
        train_times = tc[test_size+val_size:]

        test_set = combined_dataset.where(combined_dataset.time.isin(test_times) == True, drop=True)
        val_set = combined_dataset.where(combined_dataset.time.isin(val_times) == True, drop=True)
        train_set = combined_dataset.where(combined_dataset.time.isin(train_times) == True, drop=True)


        # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
        test_set.time.encoding.update(self.time_encoding)
        val_set.time.encoding.update(self.time_encoding)
        train_set.time.encoding.update(self.time_encoding)

        return {"train": train_set, "val": val_set, "test": test_set}
