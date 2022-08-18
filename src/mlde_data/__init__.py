import glob
import os

class UKCPDatasetMetadata:
    def __init__(self, base_dir, variable, frequency, domain, resolution, scenario="rcp85", ensemble_member="01", target_size=None):
        self.base_dir = base_dir
        self.variable = variable
        self.frequency = frequency
        self.resolution = resolution
        self.domain = domain
        self.target_size = target_size
        self.scenario = scenario
        self.ensemble_member = ensemble_member

        if self.resolution.startswith("2.2km"):
            self.collection = "land-cpm"
        elif self.resolution.startswith("60km"):
            self.collection = "land-gcm"

    @property
    def fq_domain(self):
        """Fully-qualified domain - for subdomains the domain is really a combination of the domain centre and its size. For full domains like 'uk' and 'global' then also need a way to bypass the size (as quite reasonably may not know it)"""
        if self.target_size is not None:
            return "-".join(self.domain, self.target_size)
        else:
            return self.domain

    def filename_prefix(self):
        return "_".join([self.variable, self.scenario, self.collection, self.fq_domain, self.resolution, self.ensemble_member, self.frequency])

    def filename(self, year):
        return f"{self.filename_prefix()}_{year-1}1201-{year}1130.nc"

    def subdir(self):
        return os.path.join(self.fq_domain, self.resolution, self.scenario, self.ensemble_member, self.variable, self.frequency)

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
