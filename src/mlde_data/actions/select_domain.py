import cartopy
import cf_xarray  # noqa: F401
import logging
import math
import numpy as np
import xarray as xr

from mlde_utils import cp_model_rotated_pole, platecarree
from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="select-subdomain")
class SelectDomain:

    osgb_crs = cartopy.crs.OSGB()

    DOMAIN_CENTRES_LON_LAT = {
        "engwales": (-1.898575, 52.489471),
        "scotland": (-4.20264580, 56.49067120),
        "uk": platecarree.transform_point(250000, 575000, src_crs=osgb_crs),
    }

    DOMAIN_CENTRES_RP_LONG_LAT = {
        domain_name: cp_model_rotated_pole.transform_point(
            lon, lat, src_crs=platecarree
        )
        + np.array([360, 0])  # for rotated pole, longitude runs over 360
        for domain_name, (lon, lat) in DOMAIN_CENTRES_LON_LAT.items()
    }

    # the way the 5km CEDA data is cut up can't use the same centres (engwales at least)
    DOMAIN_CENTRES_OSGB = {
        "engwales-5km": (372500, 287500),
        "scotland-5km": (262500, 737500),
    }

    for domain_name, (x, y) in DOMAIN_CENTRES_OSGB.items():
        DOMAIN_CENTRES_LON_LAT[domain_name] = platecarree.transform_point(
            x, y, src_crs=osgb_crs
        )

    def __init__(self, domain: str) -> None:
        self.domain = domain

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        logger.info(f"Selecting subdomain {self.domain}")
        if ds.attrs.get("domain") == f"{self.domain}":
            logger.info("Already on the desired domain, nothing to do")
            return ds

        size = self.size(ds.attrs.get("resolution"))

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

        radius_x = math.floor((size[0] - 1) / 2.0)
        radius_y = math.floor((size[1] - 1) / 2.0)
        ledge_idx = centre_long_idx - radius_x
        bedge_idx = centre_lat_idx - radius_y

        ds = ds.cf.isel(
            X=slice(ledge_idx, ledge_idx + size[0]),
            Y=slice(bedge_idx, bedge_idx + size[1]),
        )

        assert len(ds.cf["X"]) == size[0]
        assert len(ds.cf["Y"]) == size[1]

        ds = ds.assign_attrs({"domain": f"{self.domain}"})

        return ds

    def size(self, resolution: str) -> tuple[int, int]:
        if self.domain == "uk":
            if resolution == "60km" or resolution == "2.2km-coarsened-gcm":
                return (21, 21)
            else:
                raise ValueError(
                    f"Unknown resolution: {resolution} for domain: {self.domain}"
                )
        elif self.domain in ["engwales", "engwales-5km", "scotland-5km"]:
            # for target resutions, size is fixed
            if resolution == "2.2km":
                return (256, 256)
            elif resolution == "5km":
                return (128, 128)
            elif resolution == "2.2km-coarsened-4x":
                return (64, 64)
            # for GCM resolution, size is depends on domain
            elif resolution == "60km" or resolution == "2.2km-coarsened-gcm":
                if self.domain == "engwales":
                    return (13, 13)
                elif self.domain == "engwales-5km":
                    return (14, 14)
                elif self.domain == "scotland-5km":
                    return (14, 14)
                else:
                    raise ValueError(
                        f"Unknown size for domain at gcm resolution: {self.domain}"
                    )
            else:
                raise ValueError(
                    f"Unknown resolution: {resolution} for domain: {self.domain}"
                )
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
