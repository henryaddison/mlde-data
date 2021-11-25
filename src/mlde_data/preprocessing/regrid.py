import glob
import os

import iris

"""
Regrid all nc files in input directory based on the given target grid file and save output
"""
class Regrid:

    SCHEMES = {
        "linear": iris.analysis.Linear,
        "nn": iris.analysis.Nearest,
        "area-weighted": iris.analysis.AreaWeighted
    }

    def __init__(self, input_dir, target_grid, output_dir, scheme) -> None:
        self.input_dir = input_dir
        self.target_grid = target_grid
        self.output_dir = output_dir
        self.scheme = self.SCHEMES[scheme]()

        pass

    def run(self):
        output_files = []

        target_cube = iris.load_cube(str(self.target_grid))

        example_input_cube_file = glob.glob(str(self.input_dir/"*.nc"))[0]
        example_input_cube = iris.load_cube(example_input_cube_file)

        regridder = self.scheme.regridder(example_input_cube, target_cube)

        input_fileglob = str(self.input_dir/"*.nc")
        input_filepaths = glob.glob(input_fileglob)

        for input_filepath in input_filepaths:

            input_cube = iris.load_cube(input_filepath)

            single_year_regridded_cube =  regridder(input_cube)

            output_filename = os.path.basename(input_filepath)
            output_filepath = self.output_dir / output_filename
            iris.save(single_year_regridded_cube, output_filepath)
            output_files.append(output_filepath)

        print("All done")

        return output_files
