import logging

import metpy.calc as mpcalc

logger = logging.getLogger(__name__)


class Vorticity:
    def __init__(self, theta):
        self.theta = theta

    def run(self, ds):
        logger.info(f"Computing vorticity from xwind{self.theta}, ywind{self.theta}")
        if ds[f"xwind{self.theta}"].attrs["grid_mapping"] == "latitude_longitude":
            vort_da = mpcalc.vorticity(
                ds[f"xwind{self.theta}"], ds[f"ywind{self.theta}"]
            )
        elif (
            ds[f"xwind{self.theta}"].attrs["grid_mapping"]
            == "rotated_latitude_longitude"
        ):
            dx, dy = mpcalc.lat_lon_grid_deltas(
                ds.grid_longitude.values, ds.grid_latitude.values
            )
            # make sure grid deltas broadcast properly over time dimension - https://stackoverflow.com/a/55012247
            dx = dx[None, :]
            dy = dy[None, :]
            vort_da = mpcalc.vorticity(
                ds[f"xwind{self.theta}"], ds[f"ywind{self.theta}"], dx=dx, dy=dy
            )

        vort_da = vort_da.assign_attrs(
            grid_mapping=ds[f"xwind{self.theta}"].attrs["grid_mapping"],
            units="s-1",
            standard_name="atmosphere_relative_vorticity",
            long_name="relative_vorticity",
        )
        ds[f"vorticity{self.theta}"] = vort_da

        ds = ds.drop_vars([f"xwind{self.theta}", f"ywind{self.theta}"])

        return ds
