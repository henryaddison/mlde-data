import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import XRDataset

def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    return torch.load(path, map_location=device)

def predict(model, test_set, sample=None):
    if sample:
        if sample > 1:
            sample_size = sample
        else:
            sample_size = int(sample * len(test_set.time.values))
        rng = np.random.default_rng(seed=42)
        timestamps_sample = rng.choice(test_set.time, sample_size)
        test_set = test_set.where(test_set.time.isin(timestamps_sample) == True, drop=True)
    test_dl = DataLoader(XRDataset(test_set, variables=["pr"]), batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model.to(device=device)

    pred = np.concatenate([model(batch_X.to(device)).squeeze().detach().cpu().numpy() for (batch_X, _) in test_dl])

    ds = xr.Dataset(data_vars={key: test_set.data_vars[key] for key in ["time_bnds", "grid_latitude_bnds", "grid_longitude_bnds", "rotated_latitude_longitude"]}, coords=test_set.coords, attrs={})
    ds['pr'] = xr.DataArray(pred, dims=["time", "grid_latitude", "grid_longitude"])
    return ds

def open_test_set(path):
    test_set = xr.open_dataset(path)
    return test_set.assign_coords(season=(('time'), (test_set.month_number.values % 12 // 3)))
