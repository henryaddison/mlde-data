import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import XRDataset
from ml_downscaling_emulator.utils import cp_model_rotated_pole

def load_model(path):
    return torch.load(path, map_location=torch.device('cpu'))

def predict(model, test_set):
    test_dl = DataLoader(XRDataset(test_set, variables=["pr"]), batch_size=64)

    pred = np.concatenate([model(batch[0]).squeeze().detach().numpy() for batch in test_dl])

    ds = xr.Dataset(data_vars={key: test_set.data_vars[key] for key in ["time_bnds", "grid_latitude_bnds", "grid_longitude_bnds", "rotated_latitude_longitude"]}, coords=test_set.coords, attrs={})
    ds['pr'] = xr.DataArray(pred, dims=["time", "grid_latitude", "grid_longitude"])
    return ds

def open_test_set(path):
    test_set = xr.open_dataset(path)
    return test_set.assign_coords(season=(('time'), (test_set.month_number.values % 12 // 3)))

def compare_preds(models_preds, test_set, lo_res_pr_set=None):
    # lo_res_pr_set needs to be provided only if pr isn't one of the predictors in test_set
    if lo_res_pr_set is None:
        lo_res_pr_set = test_set

    models_mean_diffs = [model_preds.mean(dim=["time"]).pr - test_set.mean(dim=["time"]).target_pr for model_preds in models_preds]
    lo_res_mean_diff = (lo_res_pr_set.mean(dim=["time"]).pr - test_set.mean(dim=["time"]).target_pr)

    bias_vmin = min(lo_res_mean_diff.min().values, *[model_preds.min().values for model_preds in models_mean_diffs])
    bias_vmax = max(lo_res_mean_diff.max().values, *[model_preds.max().values for model_preds in models_mean_diffs])
    bias_vmin = min(bias_vmin, -bias_vmax)
    bias_vmax = max(bias_vmax, -bias_vmin)

    # Plot biases
    f, axes = plt.subplots(1, len(models_preds)+1, figsize=(24, 8), subplot_kw={'projection': cp_model_rotated_pole})

    for i, model_mean_diff in enumerate(models_mean_diffs):
        ax = axes[i]
        ax.coastlines()
        model_mean_diff.plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=bias_vmin, vmax=bias_vmax, cmap='BrBG')
        ax.set_title(f"Model {i} mean bias")

    ax = axes[-1]
    ax.coastlines()
    lo_res_mean_diff.plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=bias_vmin, vmax=bias_vmax, cmap='BrBG')
    ax.set_title("Lo res data mean bias")

    # Plot some examples timestamps
    f, axes = plt.subplots(1, len(models_preds)+2, figsize=(24, 8), subplot_kw={'projection': cp_model_rotated_pole})
    for i, model_preds in enumerate(models_preds):
        ax = axes[i]
        ax.coastlines()
        model_preds.pr.isel(time=0).plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=0, cmap='BrBG')
        ax.set_title(f"Model {i} precip prediction @ {test_set.time[0].values}")

    ax = axes[-2]
    ax.coastlines()
    lo_res_pr_set.pr.isel(time=0).plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=0, cmap='BrBG')
    ax.set_title(f"Lo-res precip @ {test_set.time[0].values}")
    ax = axes[-1]
    ax.coastlines()
    test_set.target_pr.isel(time=0).plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=0, cmap='BrBG')
    ax.set_title(f"Hi-res precip @ {test_set.time[0].values}")

    # Plot predicted rainfall values against hi-res ones
    f, axes = plt.subplots(1, len(models_preds)+1, figsize=(24, 8))

    tr = min(test_set.target_pr.values.max(), *[model_preds.pr.max().values for model_preds in models_preds])
    for i, model_preds in enumerate(models_preds):
        ax = axes[i]
        ax.scatter(x=test_set.target_pr, y=model_preds.pr, alpha=0.1, c='b')
        ax.plot([0, tr], [0, tr], linewidth=1, color='green')
        ax.set_title(f"Model {i} vs hi-res pr")
    ax = axes[-1]
    ax.scatter(x=test_set.target_pr, y=lo_res_pr_set.pr, alpha=0.1, c='r')
    ax.plot([0, tr], [0, tr], linewidth=1, color='green')
    ax.set_title("Lo-res pr vs hi-res pr")

    plt.figure()
    for model_preds in models_preds:
        model_preds.sum(dim=["grid_longitude", "grid_latitude"]).pr.plot.hist(alpha=0.1, log=True, density=True)
    test_set.sum(dim=["grid_longitude", "grid_latitude"]).target_pr.plot.hist(alpha=0.1, log=True, density=True)
    test_set.sum(dim=["grid_longitude", "grid_latitude"]).pr.plot.hist(alpha=0.1, log=True, density=True)
