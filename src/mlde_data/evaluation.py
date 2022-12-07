import logging

import numpy as np
import torch
import xarray as xr

def generate_samples(model, cond_batch):
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cond_batch = cond_batch.to(device)

    samples = model(cond_batch)
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # extract numpy array
    samples = samples.cpu().detach().numpy()
    return samples

def samples_to_xr(samples, xr_eval_ds, target_transform):
    coords = {**dict(xr_eval_ds.coords)}

    cf_data_vars = {key: xr_eval_ds.data_vars[key] for key in ["rotated_latitude_longitude", "time_bnds", "grid_latitude_bnds", "grid_longitude_bnds"]}

    pred_pr_dims=["time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {"grid_mapping": "rotated_latitude_longitude", "standard_name": "pred_pr", "units": "kg m-2 s-1"}
    pred_pr_var = (pred_pr_dims, samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    pred_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})

    if target_transform is not None:
        pred_ds = target_transform.invert(pred_ds)

    pred_ds = pred_ds.rename({"target_pr": "pred_pr"})

    return pred_ds

def predict(model, eval_dl, target_transform):
    preds = []
    with torch.no_grad():
        for batch_num, (cond_batch, _) in enumerate(eval_dl):
            logging.info(f"Working on batch {batch_num}")
            samples = generate_samples(model, cond_batch)
            preds.append(samples)

    # combine the samples along the time axis
    preds = np.concatenate(preds, axis=1)
    xr_data_eval  = eval_dl.dataset.ds

    preds_ds = samples_to_xr(samples, xr_data_eval, target_transform)

    return preds_ds
