import logging

import numpy as np
import torch
import xarray as xr

def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    return torch.load(path, map_location=device)

def generate_samples(model, device, cond_batch):
    cond_batch = cond_batch.to(device)

    samples = model(cond_batch)
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # add a dimension for sample_id
    samples = samples.unsqueeze(dim=0)
    # extract numpy array
    samples = samples.cpu().detach().numpy()
    return samples

def predict(model, eval_dl, target_transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    xr_data_eval  = eval_dl.dataset.ds

    preds = []

    for batch_num, (cond_batch, _) in enumerate(eval_dl):
        logging.info(f"Working on batch {batch_num}")
        samples = generate_samples(model, device, cond_batch)
        preds.append(samples)

    # combine the samples along the time axis
    preds = np.concatenate(preds, axis=1)

    # U-net is deterministic, so only 1 sample possible
    sample_id = 0

    coords = {**dict(xr_data_eval.coords), "sample_id": ("sample_id", [sample_id])}

    cf_data_vars = {key: xr_data_eval.data_vars[key] for key in ["rotated_latitude_longitude", "time_bnds", "grid_latitude_bnds", "grid_longitude_bnds", "forecast_period_bnds"]}

    pred_pr_dims=["sample_id", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {"grid_mapping": "rotated_latitude_longitude", "standard_name": "pred_pr", "units": "kg m-2 s-1"}
    pred_pr_var = (pred_pr_dims, preds, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    pred_ds = target_transform.invert(xr.Dataset(data_vars=data_vars, coords=coords, attrs={}))
    pred_ds = pred_ds.rename({"target_pr": "pred_pr"})

    return pred_ds

def open_test_set(path):
    test_set = xr.open_dataset(path)
    return test_set.assign_coords(season=(('time'), (test_set.month_number.values % 12 // 3)))
