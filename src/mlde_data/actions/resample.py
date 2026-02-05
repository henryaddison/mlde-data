import logging
import pandas as pd
import xarray as xr

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="resample")
class Resample:
    def __init__(self, frequency):
        self.frequency = frequency
        if frequency == "day":
            self.resample_kwrgs = {"time": "1D"}
            self.delta = pd.Timedelta(days=1)
        else:
            raise RuntimeError(f"Unknown target frequency {frequency}")

    def __call__(self, ds):
        logger.info(f"Resampling to {self.frequency}")

        new_bounds = ds["time_bnds"].isel(bnds=0).resample(self.resample_kwrgs).min()
        new_bounds = xr.concat([new_bounds, new_bounds + self.delta], dim="bnds")
        new_bounds = new_bounds.assign_attrs(ds["time_bnds"].attrs)
        new_bounds.encoding = ds["time_bnds"].encoding

        ds = ds.resample(**self.resample_kwrgs).mean()
        ds["time_bnds"] = new_bounds
        ds = ds.assign_attrs({"frequency": self.frequency})

        return ds
