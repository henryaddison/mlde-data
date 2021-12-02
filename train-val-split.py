import argparse
import logging
import os
from pathlib import Path

from ml_downscaling_emulator.preprocessing.train_val_split import TrainValSplit

def get_args():
    parser = argparse.ArgumentParser(description='Select a sub-region of a UKCP18 file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hi-res', dest='hi_res_files', type=Path, nargs='+', required=True,
                        help='List of hi res netcdf files')
    parser.add_argument('--lo-res', dest='lo_res_files', type=Path, nargs='+', required=True,
                        help='List of lo res netcdf files')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to store output tensors')
    parser.add_argument('--variables', dest='variables', type=str, nargs='+', required=False, default=['pr'],
                        help='List of input variables to use to create datasets')
    parser.add_argument('--val-prop', dest='val_prop', type=float, required=False, default=0.2,
                        help='Proportion of data to put in validation set')
    parser.add_argument('--test-prop', dest='test_prop', type=float, required=False, default=0.1,
                        help='Proportion of data to put in validation set')


    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    TrainValSplit(args.hi_res_files, args.lo_res_files, args.output_dir, args.variables, args.val_prop, args.test_prop).run()

    logging.info("All done")
