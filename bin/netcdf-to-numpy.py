import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr

def get_args():
    parser = argparse.ArgumentParser(description='Save a netcdf dataset to raw numpy on disk',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Base path to input storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to storage output')
    parser.add_argument('--variable', dest='variable', type=str, required=True,
                        help='Name of variable to extract and save from netCDF dataset')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    output_path = args.output_dir / f"{args.variable}.npy"

    logging.info(f"Saving {args.variable} from dataset in {args.input_dir} to {output_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    ds = xr.open_mfdataset(str(args.input_dir / "*.nc"))

    # don't need the ensemble member dimension for training
    np_array = ds.isel(ensemble_member=0)[args.variable].values

    np.save(output_path, np_array)

    logging.info("All done")