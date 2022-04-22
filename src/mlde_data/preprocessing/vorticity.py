import logging

import metpy.calc as mpcalc

logger = logging.getLogger(__name__)

class Vorticity:
    def __init__(self):
        pass

    def run(self, ds):
        logger.info(f"Computing vorticity from xwind, ywind")
        dx, dy =  mpcalc.lat_lon_grid_deltas(ds.grid_longitude.values, ds.grid_latitude.values)
        # make sure grid deltas broadcast properly over time dimension - https://stackoverflow.com/a/55012247
        dx = dx[None, :]
        dy = dy[None, :]

        ds['vorticity850'] = mpcalc.vorticity(ds['xwind'], ds['ywind'], dx=dx, dy=dy)

        ds = ds.drop_vars(['xwind', 'ywind'])

        # units='s**-1',
        # standard_name='atmosphere_relative_vorticity',
        # long_name='relative_vorticity'
        # grid_mapping='rotated_latitude_longitude'}

        return ds
