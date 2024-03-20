import logging

import numpy as np

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSplit(BaseSplit):
    def run(self, combined_dataset):
        tc = combined_dataset.time.values.copy()
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(tc)

        test_size = int(len(tc) * self.test_prop)
        val_size = int(len(tc) * self.val_prop)

        test_times = tc[0:test_size]
        val_times = tc[test_size : test_size + val_size]
        train_times = tc[test_size + val_size :]

        test_set = combined_dataset.where(
            combined_dataset.time.isin(test_times) == True, drop=True  # noqa: E712
        )
        val_set = combined_dataset.where(
            combined_dataset.time.isin(val_times) == True, drop=True  # noqa: E712
        )
        train_set = combined_dataset.where(
            combined_dataset.time.isin(train_times) == True, drop=True  # noqa: E712
        )

        return {"train": train_set, "val": val_set, "test": test_set}
