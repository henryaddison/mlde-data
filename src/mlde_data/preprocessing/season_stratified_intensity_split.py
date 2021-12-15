import datetime
import logging
import os

import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
import xarray as xr

logger = logging.getLogger(__name__)

class SeasonStratifiedIntensitySplit:
    def __init__(self, lo_res_files, hi_res_files, output_dir, variables, val_prop=0.2, test_prop=0.1) -> None:
        self.lo_res_files = lo_res_files
        self.hi_res_files = hi_res_files
        self.variables = variables
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.output_dir = output_dir

    def run(self):
        time_encoding = xr.open_dataset(self.lo_res_files[0]).time_bnds.encoding
        lo_res_dataset = xr.open_mfdataset(self.lo_res_files)
        hi_res_dataset = xr.open_mfdataset(self.hi_res_files).rename({'pr': 'target_pr', 'ensemble_member_id': 'cpm_ensemble_member_id'})

        combined_dataset = xr.combine_by_coords([lo_res_dataset, hi_res_dataset], compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all").isel(ensemble_member=0)
        combined_dataset = combined_dataset.assign_coords(season=(('time'), (combined_dataset.month_number.values % 12 // 3)))

        test_times = set()
        val_times = set()

        for iseason, group in combined_dataset.groupby("season"):
            logger.debug(f"Working on season {iseason}")
            season_sorted_time = np.flipud(group.sum(dim=['grid_latitude', 'grid_longitude']).sortby('target_pr').time.values.copy())

            season_time_set = set(season_sorted_time)

            season_test_size = int(len(season_sorted_time)*self.test_prop)
            season_val_size = int(len(season_sorted_time)*self.val_prop)

            season_test_times = set()
            season_val_times = set()

            working_set = season_test_times

            for timestamp in season_sorted_time:
                lag_duration = 5
                event = set(np.arange(timestamp - datetime.timedelta(days = lag_duration), timestamp + datetime.timedelta(days = lag_duration+1), datetime.timedelta(days = 1), dtype=type(timestamp))) & season_time_set
                unseen_event_days = event - test_times - val_times
                working_set.update(unseen_event_days)
                if len(season_test_times) >= season_test_size:
                    working_set = season_val_times
                if len(season_val_times) >= season_val_size:
                    break

            test_times |= season_test_times
            val_times |= season_val_times

        train_times = set(combined_dataset.time.values) - test_times - val_times

        test_set = combined_dataset.where(combined_dataset.time.isin(list(test_times)) == True, drop=True)
        val_set = combined_dataset.where(combined_dataset.time.isin(list(val_times)) == True, drop=True)
        train_set = combined_dataset.where(combined_dataset.time.isin(list(train_times)) == True, drop=True)

        # # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
        test_set.time.encoding.update(time_encoding)
        val_set.time.encoding.update(time_encoding)
        train_set.time.encoding.update(time_encoding)

        logger.info(f"Saving data to {self.output_dir}")
        test_set.to_netcdf(os.path.join(self.output_dir, 'test.nc'))
        val_set.to_netcdf(os.path.join(self.output_dir, 'val.nc'))
        train_set.to_netcdf(os.path.join(self.output_dir, 'train.nc'))
