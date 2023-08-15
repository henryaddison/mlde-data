"""
Change longitude from 0-360 to -180 to 180
so break doesn't appear over UK
"""


class ShiftLonBreak:
    def __init__(self, lon_name="longitude") -> None:
        self.lon_name = lon_name

    def run(self, ds):
        orig_lon_attrs = ds[self.lon_name].attrs
        ds.coords[self.lon_name] = (ds.coords[self.lon_name] + 180) % 360 - 180
        ds = ds.sortby(ds[self.lon_name])
        ds[self.lon_name].attrs = orig_lon_attrs

        return ds
