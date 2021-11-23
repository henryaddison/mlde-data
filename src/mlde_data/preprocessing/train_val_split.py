import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import xarray as xr

class TrainValSplit:
    def __init__(self, lo_res_files, hi_res_files, output_dir, variables = ['pr'], val_prop=0.3) -> None:
        self.lo_res_files = lo_res_files
        self.hi_res_files = hi_res_files
        self.variables = variables
        self.val_prop = val_prop
        self.output_dir = output_dir

    def run(self):
        gcm_dataset = xr.open_mfdataset(self.lo_res_files).isel(ensemble_member=0)
        cpm_dataset = xr.open_mfdataset(self.hi_res_files).isel(ensemble_member=0).rename({'pr': 'target_pr', 'ensemble_member_id': 'cpm_ensemble_member_id'})

        combined_dataset = xr.merge([gcm_dataset, cpm_dataset], join='inner')

        unstacked_X = [torch.tensor(combined_dataset[variable].values) for variable in self.variables]

        X = torch.stack(list(unstacked_X), dim=1)
        y = torch.tensor(combined_dataset['target_pr'].values).unsqueeze(dim=1)

        all_data = TensorDataset(X, y)

        val_size = int(self.val_prop * len(all_data))
        train_size = len(all_data) - val_size
        train_set, val_set = random_split(all_data, [train_size, val_size])

        train_X, train_y = train_set[:]
        val_X, val_y = val_set[:]

        torch.save(train_X, self.output_dir/'train_X.pt')
        torch.save(train_y, self.output_dir/'train_y.pt')

        torch.save(val_X, self.output_dir/'val_X.pt')
        torch.save(val_y, self.output_dir/'val_y.pt')
