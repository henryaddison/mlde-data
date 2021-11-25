import argparse
import logging
import os
from pathlib import Path

from ml_downscaling_emulator.preprocessing.select_subregion import SelectSubregion

def get_args():
    parser = argparse.ArgumentParser(description='Select a sub-region of a UKCP18 file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input_dir', type=Path, required=True,
                        help='Base path to input storage')
    parser.add_argument('--output', dest='output_dir', type=Path, required=True,
                        help='Base path to storage output')

    subregion_defn_group = parser.add_mutually_exclusive_group()
    subregion_defn_group.add_argument("--london", dest="subregion_defn" , action="store_const", const=SelectSubregion.LONDON_IN_CPM)
    subregion_defn_group.add_argument("--london-gcm", dest="subregion_defn", action="store_const", const=SelectSubregion.LONDON_IN_GCM)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s: %(message)s')

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outputs = SelectSubregion(args.input_dir, args.output_dir, args.subregion_defn).run()

    logging.info(outputs)
