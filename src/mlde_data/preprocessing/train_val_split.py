import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import xarray as xr

class TrainValSplit:
    def __init__(self, lo_res_files, hi_res_files, output_dir, variables = ['pr'], val_prop=0.2, test_prop=0.1) -> None:
        self.lo_res_files = lo_res_files
        self.hi_res_files = hi_res_files
        self.variables = variables
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.output_dir = output_dir

    def run(self):
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
        test_set.time.encoding.update(lo_res_dataset.time_bnds.encoding)
        val_set.time.encoding.update(lo_res_dataset.time_bnds.encoding)
        train_set.time.encoding.update(lo_res_dataset.time_bnds.encoding)

        test_set.to_netcdf(self.output_dir/'test.nc')
        val_set.to_netcdf(self.output_dir/'val.nc')
        train_set.to_netcdf(self.output_dir/'train.nc')


        # unstacked_X = [torch.tensor(combined_dataset[variable].values) for variable in self.variables]

        # X = torch.stack(list(unstacked_X), dim=1)
        # y = torch.tensor(combined_dataset['target_pr'].values).unsqueeze(dim=1)

        # all_data = TensorDataset(X, y)

        # val_size = int(self.val_prop * len(all_data))
        # train_size = len(all_data) - val_size
        # train_set, val_set = random_split(all_data, [train_size, val_size])

        # train_X, train_y = train_set[:]
        # val_X, val_y = val_set[:]

        # torch.save(train_X, self.output_dir/'train_X.pt')
        # torch.save(train_y, self.output_dir/'train_y.pt')

        # torch.save(val_X, self.output_dir/'val_X.pt')
        # torch.save(val_y, self.output_dir/'val_y.pt')
