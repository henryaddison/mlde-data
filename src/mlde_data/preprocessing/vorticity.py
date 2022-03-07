from windspharm.xarray import to3d, get_apiorder, inspect_gridtype
import windspharm
import xarray as xr

class Vorticity:
    def __init__(self):
        self.gridtype = 'regular'

    def run(self, ds):
        u = ds['xwind']
        v = ds['ywind']

        lat = u.coords['grid_latitude']
        lat_dim = u.dims.index(lat.name)

        lon = u.coords['grid_longitude']
        lon_dim = u.dims.index(lon.name)

        if lat.values[0] < lat.values[1]:
            u = self._reverse(u, lat_dim)
            v = self._reverse(v, lat_dim)

            lat = u.coords['grid_latitude']
            lat_dim = u.dims.index(lat.name)

        apiorder, _ = get_apiorder(u.ndim, lat_dim, lon_dim)
        apiorder = [u.dims[i] for i in apiorder]

        reorder = u.dims
        u = u.copy().transpose(*apiorder)
        v = v.copy().transpose(*apiorder)
        # Reshape the raw data and input into the API.
        ishape = u.shape
        coords = [u.coords[name] for name in u.dims]

        u = to3d(u.values)
        v = to3d(v.values)

        api = windspharm.standard.VectorWind(u, v, gridtype=self.gridtype)

        vrt = api.vorticity()

        ds['vorticity'] = self._metadata(vrt, 'vorticity', ishape, coords, reorder,
            units='s**-1',
            standard_name='atmosphere_relative_vorticity',
            long_name='relative_vorticity')

        return ds

    def _reverse(self, array, dim):
        """Reverse an `xarray.DataArray` along a given dimension."""
        slicers = [slice(0, None)] * array.ndim
        slicers[dim] = slice(-1, None, -1)
        return array[tuple(slicers)]

    def _metadata(self, var, name, ishape, coords, reorder, **attributes):
        var = var.reshape(ishape)
        var = xr.DataArray(var, coords=coords, name=name)
        var = var.transpose(*reorder)
        for attr, value in attributes.items():
            var.attrs[attr] = value
        return var