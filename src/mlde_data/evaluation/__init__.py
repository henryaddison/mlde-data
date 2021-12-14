import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import XRDataset
from ml_downscaling_emulator.utils import cp_model_rotated_pole
from ml_downscaling_emulator.helpers import plots_at_ts

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

def plot_mean_diffs(models_mean_diffs, model_names):
    bias_vmin = min(*[diff.pr.min().values for diff in models_mean_diffs])
    bias_vmax = max(*[diff.pr.max().values for diff in models_mean_diffs])
    bias_vmin = min(bias_vmin, -bias_vmax)
    bias_vmax = max(bias_vmax, -bias_vmin)

    f, axes = plt.subplots(1, len(models_mean_diffs), figsize=(24, 8), subplot_kw={'projection': cp_model_rotated_pole})

    for i, model_mean_diff in enumerate(models_mean_diffs):
        ax = axes[i]
        ax.coastlines()
        model_mean_diff.pr.plot(ax = ax, x='grid_longitude', y='grid_latitude', transform=cp_model_rotated_pole, vmin=bias_vmin, vmax=bias_vmax, cmap='BrBG')
        ax.set_title(f"{model_names[i]} mean bias")

def compare_preds(models_preds, models_mean_diffs, test_set, model_names):
    plot_mean_diffs(models_mean_diffs, model_names)

    # lo_res_pr_set needs to be provided only if pr isn't one of the predictors in test_set

    timestamp = test_set.time[0].values
    raw_pr = [test_set.target_pr]+[pred.pr for pred in models_preds]
    plots_at_ts(raw_pr, timestamp, titles=["Target pr from CPM"]+model_names)#, vmax=vmax)
    residual_pr = [pred - test_set.target_pr for pred in raw_pr]
    plots_at_ts(residual_pr, timestamp, titles=["Target pr from CPM"]+model_names, vmin=None, cmap="BrBG")#, vmax=vmax)


    # Plot predicted rainfall values against hi-res ones
    f, axes = plt.subplots(1, len(models_preds), figsize=(24, 8))

    tr = min(test_set.target_pr.values.max(), *[model_preds.pr.max().values for model_preds in models_preds])
    for i, model_preds in enumerate(models_preds):
        ax = axes[i]
        ax.scatter(x=test_set.target_pr, y=model_preds.pr, alpha=0.1, c='b')
        ax.plot([0, tr], [0, tr], linewidth=1, color='green')
        ax.set_title(f"{model_names[i]} predicted pr vs hi-res target pr")

    plt.figure()
    for model_preds in models_preds:
        model_preds.sum(dim=["grid_longitude", "grid_latitude"]).pr.plot.hist(alpha=0.1, log=True, density=True)
    test_set.sum(dim=["grid_longitude", "grid_latitude"]).target_pr.plot.hist(alpha=0.1, log=True, density=True)
