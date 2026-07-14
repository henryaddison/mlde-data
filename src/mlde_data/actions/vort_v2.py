import logging

import metpy.calc as mpcalc

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="vort-v2")
class VortV2:
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, ds):
        logger.info(f"Computing vorticity @ {self.theta} from x_wind, y_wind")
        if ds["x_wind"].attrs["grid_mapping"] == "latitude_longitude":
            ds = ds.metpy.parse_cf()
            # ds["x_wind"] = ds["x_wind"].metpy.assign_crs(
            #     grid_mapping_name=ds["latitude_longitude"].attrs["grid_mapping_name"],
            #     earth_radius=ds["latitude_longitude"].attrs["earth_radius"],
            # )
            # ds["y_wind"] = ds["y_wind"].metpy.assign_crs(
            #     grid_mapping_name=ds["latitude_longitude"].attrs["grid_mapping_name"],
            #     earth_radius=ds["latitude_longitude"].attrs["earth_radius"],
            # )
            vort_da = mpcalc.vorticity(ds["x_wind"], ds["y_wind"])
        else:
            raise RuntimeError(
                f"Unrecognised grid mapping {ds['x_wind'].attrs['grid_mapping']} for vorticity calculation"
            )

        vort_da = vort_da.assign_attrs(
            grid_mapping=ds[f"x_wind"].attrs["grid_mapping"],
            units="s-1",
            standard_name="atmosphere_relative_vorticity",
            long_name="relative_vorticity",
        )
        ds[f"vort{self.theta}"] = vort_da
        ds = ds.reset_coords("metpy_crs", drop=True)
        # vort_da

        return ds
