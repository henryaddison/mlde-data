import logging
import os

import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import xarray as xr

logger = logging.getLogger(__name__)

class TrainValSplit:
    def __init__(self, lo_res_files, hi_res_files, output_dir, variables = ['pr'], val_prop=0.2, test_prop=0.1) -> None:
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

        tc = combined_dataset.time.values.copy()
        np.random.shuffle(tc)

        test_size = int(len(tc)*self.test_prop)
        val_size = int(len(tc)*self.val_prop)

        test_times = tc[0:test_size]
        val_times = tc[test_size:test_size+val_size]
        train_times = tc[test_size+val_size:]

        test_set = combined_dataset.where(combined_dataset.time.isin(test_times) == True, drop=True)
        val_set = combined_dataset.where(combined_dataset.time.isin(val_times) == True, drop=True)
        train_set = combined_dataset.where(combined_dataset.time.isin(train_times) == True, drop=True)

        # https://github.com/pydata/xarray/issues/2436 - time dim encoding lost when opened using open_mfdataset
        test_set.time.encoding.update(time_encoding)
        val_set.time.encoding.update(time_encoding)
        train_set.time.encoding.update(time_encoding)

        logger.info(f"Saving data to {self.output_dir}")
        self.save(test_set, 'test')
        self.save(val_set, 'val')
        self.save(train_set, 'train')

    def save(self, ds, label):
        ds.to_netcdf(os.path.join(self.output_dir, f"{label}.nc"))
        unstacked_X = [torch.tensor(ds[variable].values) for variable in self.variables]

        X = torch.stack(list(unstacked_X), dim=1)
        y = torch.tensor(ds['target_pr'].values).unsqueeze(dim=1)

        torch.save(X, os.path.join(self.output_dir, f"{label}_X.pt"))
        torch.save(X, os.path.join(self.output_dir, f"{label}_y.pt"))
