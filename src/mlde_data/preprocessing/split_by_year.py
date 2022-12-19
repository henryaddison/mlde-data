import itertools

import cftime
import xarray

"""
Split up an nc multifile dataset into single file per year.
"""


class SplitByYear:
    def __init__(
        self,
        input_filepath_prefix,
        output_filepath_prefix,
        years=itertools.chain(range(1980, 2000), range(2020, 2040), range(2060, 2080)),
    ) -> None:
        self.input_filepath_prefix = input_filepath_prefix
        self.output_filepath_prefix = output_filepath_prefix
        self.years = years

        pass

    def gcm_file_year_range(self, year):
        if (year % 10) <= 8:
            start = (year // 10) * 10 - 1
        else:
            start = year

        end = start + 10

        return f"{start}1201-{end}1130"

    def run(self):
        output_files = []

        for year in self.years:
            input_filepath = (
                f"{self.input_filepath_prefix}_{self.gcm_file_year_range(year)}.nc"
            )
            print(f"Opening {input_filepath}")
            input = xarray.load_dataset(input_filepath)
            single_year_input = input.sel(
                time=slice(
                    cftime.Datetime360Day(year, 12, 1, 12, 0, 0, 0),
                    cftime.Datetime360Day(year + 1, 11, 30, 12, 0, 0, 0),
                )
            )

            output_filepath = (
                f"{self.output_filepath_prefix}_{year}1201-{year+1}1130.nc"
            )
            single_year_input.to_netcdf(output_filepath)
            output_files.append(output_filepath)

        print("All done")

        return output_files
