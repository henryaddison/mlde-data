import logging

import metpy.calc as mpcalc

logger = logging.getLogger(__name__)

class Vorticity:
    def __init__(self):
        pass

    def run(self, ds):
        logger.info(f"Computing vorticity from xwind, ywind")
        if ds['xwind'].attrs['grid_mapping'] == "latitude_longitude":
            vort_da = mpcalc.vorticity(ds['xwind'], ds['ywind'])
        elif ds['xwind'].attrs['grid_mapping'] == "rotated_latitude_longitude":
            dx, dy =  mpcalc.lat_lon_grid_deltas(ds.grid_longitude.values, ds.grid_latitude.values)
            # make sure grid deltas broadcast properly over time dimension - https://stackoverflow.com/a/55012247
            dx = dx[None, :]
            dy = dy[None, :]
            vort_da = mpcalc.vorticity(ds['xwind'], ds['ywind'], dx=dx, dy=dy)

        vort_da = vort_da.assign_attrs(grid_mapping=ds['xwind'].attrs['grid_mapping'], units="s-1", standard_name="atmosphere_relative_vorticity", long_name="relative_vorticity")
        ds['vorticity850'] = vort_da

        ds = ds.drop_vars(['xwind', 'ywind'])

        return ds
