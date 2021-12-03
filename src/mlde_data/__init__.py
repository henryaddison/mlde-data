import glob
import os

class UKCPDatasetMetadata:
    def __init__(self, base_dir, domain, resolution, scenario, ensemble_member, variable, frequency):
        self.base_dir = base_dir
        self.domain = domain
        self.resolution = resolution
        self.scenario = scenario
        self.ensemble_member = ensemble_member
        self.variable = variable
        self.frequency = frequency

        if self.resolution.startswith("2.2km"):
            self.collection = "land-cpm"
            self.filename_resolution = "2.2km"
        elif self.resolution.startswith("60km"):
            self.collection = "land-gcm"
            self.filename_resolution = "60km"

    def filename_prefix(self):
        return "_".join([self.variable, self.scenario, self.collection, "uk", self.filename_resolution, self.ensemble_member, self.frequency])

    def filename(self, year):
        return f"{self.filename_prefix()}_{year}1201-{year+1}1130.nc"

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
