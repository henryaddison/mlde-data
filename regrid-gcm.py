import argparse
import os
from pathlib import Path

from ml_downscaling_emulator.preprocessing.regrid import Regrid

def get_args():
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Path to directory holding raw files')
    parser.add_argument('--target', dest='target_file', type=Path, required=True,
                        help='Base path to raw data storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Path to directory to store regridded files')
    parser.add_argument('--scheme', dest='regrid_scheme', type=str, required=False,
                        choices=Regrid.SCHEMES.keys(), default='linear',
                        help='Regridding scheme')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outputs = Regrid(args.input_dir, args.target_file, args.output_dir, args.regrid_scheme).run()

    print(outputs)
