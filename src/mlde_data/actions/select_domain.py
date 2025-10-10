import cartopy
import cf_xarray  # noqa: F401
import logging
import math
import numpy as np

from mlde_utils import cp_model_rotated_pole, platecarree
from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="select-subdomain")
class SelectDomain:

    DOMAIN_CENTRES_LON_LAT = {
        "london": (-0.118092, 51.509865),
        "birmingham": (-1.898575, 52.489471),
        "glasgow": (-4.25763000, 55.86515000),
        "aberdeen": (-2.09814000, 57.14369000),
        "scotland": (-4.20264580, 56.49067120),
        "dublin": (-6.267494, 53.344105),
    }

    DOMAIN_CENTRES_RP_LONG_LAT = {
        domain_name: cp_model_rotated_pole.transform_point(
            lon, lat, src_crs=platecarree
        )
        + np.array([360, 0])  # for rotated pole, longitude runs over 360
        for domain_name, (lon, lat) in DOMAIN_CENTRES_LON_LAT.items()
    }

    osgb_crs = cartopy.crs.OSGB()

    DOMAIN_CENTRES_OSGB = {
        domain_name: cartopy.crs.OSGB().transform_point(lon, lat, src_crs=platecarree)
        for domain_name, (lon, lat) in DOMAIN_CENTRES_LON_LAT.items()
    }

    def __init__(self, domain, size=64) -> None:
        self.domain = domain
        self.size = size

    def __call__(self, ds):
        logger.info(f"Selecting subdomain {self.domain}")
        if "rotated_latitude_longitude" in ds.cf.grid_mapping_names:
            centre_xy = self.DOMAIN_CENTRES_RP_LONG_LAT[self.domain]
            query = dict(
                X=centre_xy[0],
                Y=centre_xy[1],
            )
        elif "latitude_longitude" in ds.cf.grid_mapping_names:
            centre_xy = self.DOMAIN_CENTRES_LON_LAT[self.domain]
            query = dict(
                X=centre_xy[0],
                Y=centre_xy[1],
            )
        elif "transverse_mercator" in ds.cf.grid_mapping_names:
            centre_xy = self.DOMAIN_CENTRES_OSGB[self.domain]
            query = dict(
                X=centre_xy[0],
                Y=centre_xy[1],
            )
        else:
            raise ValueError(f"Unknown grid type: {self.grid}")

        centre_ds = ds.cf.sel(query, method="nearest")
        centre_long_idx = np.where(ds.cf["X"].values == centre_ds.cf["X"].values)[
            0
        ].item()
        centre_lat_idx = np.where(ds.cf["Y"].values == centre_ds.cf["Y"].values)[
            0
        ].item()

        radius = math.floor((self.size - 1) / 2.0)
        ledge_idx = centre_long_idx - radius
        bedge_idx = centre_lat_idx - radius

        ds = ds.cf.isel(
            X=slice(ledge_idx, ledge_idx + self.size),
            Y=slice(bedge_idx, bedge_idx + self.size),
        )

        assert len(ds.cf["X"]) == self.size
        assert len(ds.cf["Y"]) == self.size

        ds = ds.assign_attrs({"domain": f"{self.domain}-{self.size}"})

        return ds
