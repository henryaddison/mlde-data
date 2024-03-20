import datetime
import logging

import numpy as np

from .base_split import BaseSplit
from .random_split import RandomSplit

logger = logging.getLogger(__name__)


class SeasonStratifiedIntensitySplit(BaseSplit):
    def run(self, combined_dataset):
        test_times = set()
        val_times = set()

        for iseason, group in combined_dataset.groupby("season"):
            logger.debug(f"Working on season {iseason}")
            season_sorted_time = np.flipud(
                group.sum(dim=["grid_latitude", "grid_longitude"])
                .sortby("target_pr")
                .time.values.copy()
            )

            season_time_set = set(season_sorted_time)

            season_test_size = int(len(season_sorted_time) * self.test_prop)
            season_val_size = int(len(season_sorted_time) * self.val_prop)

            season_test_times = set()
            season_val_times = set()

            working_set = season_test_times

            for timestamp in season_sorted_time:
                lag_duration = 5
                event = (
                    set(
                        np.arange(
                            timestamp - datetime.timedelta(days=lag_duration),
                            timestamp + datetime.timedelta(days=lag_duration + 1),
                            datetime.timedelta(days=1),
                            dtype=type(timestamp),
                        )
                    )
                    & season_time_set
                )
                unseen_event_days = event - test_times - val_times
                working_set.update(unseen_event_days)
                if len(season_test_times) >= season_test_size:
                    working_set = season_val_times
                if len(season_val_times) >= season_val_size:
                    break

            test_times |= season_test_times
            val_times |= season_val_times

        train_times = set(combined_dataset.time.values) - test_times - val_times

        extreme_test_set = combined_dataset.where(
            combined_dataset.time.isin(list(test_times)) == True,  # noqa: E712
            drop=True,
        )
        extreme_val_set = combined_dataset.where(
            combined_dataset.time.isin(list(val_times)) == True, drop=True  # noqa: E712
        )
        extreme_train_set = combined_dataset.where(
            combined_dataset.time.isin(list(train_times)) == True,  # noqa: E712
            drop=True,
        )

        splits = RandomSplit(
            val_prop=self.val_prop,
            test_prop=self.test_prop,
            seed=self.seed,
        ).run(extreme_train_set)
        splits.update(
            {"extreme_val": extreme_val_set, "extreme_test": extreme_test_set}
        )

        return splits
