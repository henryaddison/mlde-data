import cf_xarray  # noqa: F401
import logging
import xarray as xr

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="trim-domain")
class TrimDomain:

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        logger.info(f"Trimming domain to be a multiple of {self.patch_size}...")

        ds = ds.isel(
            {
                ds.cf["X"].name: slice(ds.cf["X"].size % self.patch_size, None),
                ds.cf["Y"].name: slice(ds.cf["Y"].size % self.patch_size, None),
            }
        )

        return ds
