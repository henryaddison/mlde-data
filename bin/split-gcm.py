import argparse
import os
from pathlib import Path

from ml_downscaling_emulator.preprocessing.split_by_year import SplitByYear

def get_args():
    parser = argparse.ArgumentParser(description='Regrid GCM data to match the CPM data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-prefix', dest='input_prefix', type=Path, required=True,
                        help='Prefix of input files to split up (so filepath up to the date part)')
    parser.add_argument('--output-prefix', dest='output_prefix', type=str, required=True,
                        help='Prefix of output files including directory path')
    parser.add_argument('--years', dest='years', nargs='+', type=int, required=True,
                        help='Years to cover')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    outputs = SplitByYear(args.input_prefix, args.output_prefix, args.years).run()

    print(outputs)
