# ML Dowscaling Emulation Data

The purpose of this application is to extract climate model variables from the Met Office's storage system and to manage datasets for the ML emulator.

Extracted variables are stored in separate netCDF files per variable, split up by time (currently per year, 1st December to 30th November).
An extracted variable has time, x and y dimensions. For multi-level variables (e.g. varying altitude/pressure levels), this is encoded in the variable name.
Other properties are scenario (currently only RCP8.5), spatial domain and resolution, ensemble member (01, 04-13 & 15).
Spatial resolution has two parts: the resolution of the data and the grid resolution stored at in the file. These might not be the same - e.g. 60km GCM data is stored at the targeted 4x coarsened CPM resolution (i.e. 8.8km on the CPM rotated pole grid mapping) to make it easy to combine data from different resolutions (they have the same x and y dimensions and coordinates and grid mapping in netCDF). Each part of these resolutions may be broken down further: the source resolution and any remapping (e.g. 2.2km-coarsened-gcm means 2.2km data from CPM which has been remapped onto the 60km GCM grid, 2.2km-coarsened-4x means 2.2km data coarsened 4x in x and y axis).
60km resolution is synonamous with the GCM grid and 2.2km resolution is synonamous with the CPM grid.

A dataset is a combination of many variables over many years. They consist of 3 netCDF files (train.nc, test.nc and val.nc) and a YML ds-config.yml file. To combine the variables they must have:
* cover the same spatial domain at the same resolution, i.e. same grid-mapping, grid resolution, scenario, centre and size OR in netCDF/xarray terms have the same coordinates for the same x and y dimensions (i.e. `grid_latitude` and `grid_longitude` on the Met Office's UK CPM rotated pole grid, or `latitude` and `longitude` on the Met Office's GCM grid).
* cover the same ensemble members.
* cover the same temporal domain at the same frequency: the same coordinates for the `time` dimension. This is assumed to be daily data for years 1981-2000, 2021-2040 and 2061-2080 (this is typically daily means but this isn't a hard constraint, merely 1 value per day is assumed).
* follow the same scenario.

The resolution, variables, scenario, ensemble members are defined in the config file. As is the scheme for splitting the data across the three subsets.

Note that predictor and predictand variables may have different data resolutions - they just need to have been regridded to the same grid resolution. Indeed the purpose of this is to combine coarse predictor variables with fine predictand variables.

### File paths

`${DERIVED_DATA}/moose` for variables.

`$DERIVED_DATA/moose/nc-datasets` for datasets.

## Installation

Assumes you have conda (or equivalent like mamba installed).

1. Clone repo and cd into it
2. Create conda environment: `conda env create -f environment.lock.yml` (or add dependencies to your own: `conda install --file=environment.txt`)
3. Activate the conda environment (if not already done so)
4. Install package: `pip install -e .`
5. Configure application behaviour with environment variables. See `.env.example` for variables that can be set.

### Updating conda environment

To add new packages or update their version, it is recommended to use the `environment.txt` file (for conda packages) and `requirements.txt` file (for pip packages) then run:
```sh
conda install --file=environment.txt
pip install -e . # this will implicitly use requirement.txt
conda env export -f environment.lock.yml
```
then commit any changes (though make sure not to include mlde-notebooks package in the lock file since that is not distributed via PyPI).

To sync environment with the lock file use:
```sh
conda env update -f environment.lock.yml --prune
```

## Usage

### Creating variables

See example scripts in `bin/moose/` for extracting and transforming data from Moose on JASMIN into variable files on University of Bristol's Blue Pebble HPC system.

### Creating datasets

Once you have extracted the variable files, use the `mlde-data dataset create` command to create a dataset from them ready for the machine learning code.

Example usage:

```sh
mlde-data dataset create src/mlde_data/config/datasets/bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr.yml
```

This will create the files for a dataset in `${DERIVED_DATA}/moose/nc-datasets/bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr` based on the config file supplied.

### Validation

Variables and datasets can be validated with `mlde-data variable validate` and `mlde-data dataset validate` commands respectively.
