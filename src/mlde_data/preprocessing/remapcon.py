import tempfile
from cdo import Cdo

class Remapcon:
    def __init__(self, target_grid_filepath):
        self.target_grid_filepath = target_grid_filepath

    def run(self, ds):
        input_file = tempfile.NamedTemporaryFile(delete=True, prefix='cdo_xr_input_', dir=tempfile.gettempdir())
        print(input_file.name)
        ds.to_netcdf(input_file.name)

        cdo = Cdo()
        return cdo.remapcon(self.target_grid_filepath, input=input_file.name, returnXDataset = True)
