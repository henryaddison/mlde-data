import glob
import os

class UKCPDatasetMetadata:
    def __init__(self, base_dir, variable, frequency, domain, resolution, scenario="rcp85", ensemble_member="01"):
        self.base_dir = base_dir
        self.variable = variable
        self.frequency = frequency
        self.resolution = resolution
        self.domain = domain
        self.scenario = scenario
        self.ensemble_member = ensemble_member

        if self.resolution.startswith("2.2km"):
            self.collection = "land-cpm"
        elif self.resolution.startswith("60km"):
            self.collection = "land-gcm"

    def filename_prefix(self):
        return "_".join([self.variable, self.scenario, self.collection, self.domain, self.resolution, self.ensemble_member, self.frequency])

    def filename(self, year):
        return f"{self.filename_prefix()}_{year-1}1201-{year}1130.nc"

    def subdir(self):
        return os.path.join(self.domain, self.resolution, self.scenario, self.ensemble_member, self.variable, self.frequency)

    def dirpath(self):
        return os.path.join(self.base_dir, self.subdir())

    def filepath(self, year):
        return os.path.join(self.dirpath(), self.filename(year))

    def filepath_prefix(self):
        return os.path.join(self.dirpath(), self.filename_prefix())

    def existing_filepaths(self):
        return glob.glob(f"{self.filepath_prefix()}_*.nc")

    def years(self):
        filenames = [os.path.basename(filepath) for filepath in self.existing_filepaths()]
        return list([int(filename[-20:-16]) for filename in filenames])
