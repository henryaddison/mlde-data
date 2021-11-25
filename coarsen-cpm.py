import argparse
import logging
import os
from pathlib import Path

from ml_downscaling_emulator.preprocessing.coarsen import Coarsen

def get_args():
    parser = argparse.ArgumentParser(description='Coarsen a diectory hi-res files to a lower resolution',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Base path to input storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to storage output')
    parser.add_argument('--scale-factor', dest="scale_factor", type=int, default=4, help='Scale factor for coarsening')
    parser.add_argument('--variable', type=str, default="pr", help="Data variable to re-scale")
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outputs = Coarsen(args.input_dir, args.output_dir, args.scale_factor, args.variable).run()

    logging.info(outputs)
