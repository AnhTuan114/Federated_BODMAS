"""
Transform PE samples from original format to non-executable format.

    @requirements: python3, pandas, pefile python module
​
    @input: filelist - a local file that needs to be constructed containing
            paths of each sample One per line.
    @input: altered_folder - path/name to a local directory where to save
            transformed executables.
    @input: arm - Determines if the target file should be armed or disarmed
​
    @return: None. Samples should be transformed and uploaded to provided S3
            bucket. Otherwise modify code to keep local transformed samples (no clean up)
"""

import pefile
from pefile import PE
import argparse
import logging
import hashlib
import os
import traceback
import pandas as pd
from timeit import default_timer as timer


logger = logging.getLogger()
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s] %(message)s",
            handlers=[
                 logging.StreamHandler()
            ])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", action="store", required=True)
    parser.add_argument("--arm", action="store_true", default=False)
    parser.add_argument("--save_record", action="store", default='meta_disarm.csv')
    parser.add_argument("--altered_folder")
    args = parser.parse_args()
    return args


def hash_file(file_path):
    return hashlib.sha256(open(file_path, 'rb').read()).hexdigest()


def read_hash_from_filename(file_path):
    return os.path.basename(file_path).replace('.exe', '')


def alter_pe_file(file_path, out_path, val1, val2, record_file=None):
    pe = PE(file_path)
    if record_file is not None:
        old_val1 = pe.OPTIONAL_HEADER.Subsystem
        old_val2 = pe.FILE_HEADER.Machine
        hash = read_hash_from_filename(file_path)
        with open(record_file, 'a') as f:
            f.write(f'{hash},{old_val1},{old_val2}\n')
    pe.OPTIONAL_HEADER.Subsystem = val1
    pe.FILE_HEADER.Machine = val2

    pe.write(filename=out_path)


def arm_pe_file(file_path, altered_file, sha_value_dict):
    sha = read_hash_from_filename(file_path)
    v = sha_value_dict['OPTIONAL_HEADER.Subsystem'][sha]
    z = sha_value_dict['FILE_HEADER.Machine'][sha]
    alter_pe_file(file_path, altered_file, v, z)
    return True


def disarm_pe_file(file_path, altered_file, record_file):
    alter_pe_file(file_path, altered_file, 0, 0, record_file)
    return True


def init_header(record_file):
    with open(record_file, 'w') as f:
        f.write('sha256,OPTIONAL_HEADER.Subsystem,FILE_HEADER.Machine\n')


def main(args):
    files = None
    with open(args.filelist) as fd:
        files = fd.read().splitlines()

    record_file = args.save_record
    if not os.path.exists(record_file):
        init_header(record_file)
    else:
        df = pd.read_csv(record_file, index_col='sha256')
        logger.info(f"Loaded sha_value_dict: {df}")
        sha_value_dict = df.to_dict()
        logger.info(f"Loaded sha_value_dict: {sha_value_dict}")

    for idx, file in enumerate(files):
        logger.info(f'{idx+1} Started processing file {file}')

        altered_file = os.path.join(args.altered_folder, os.path.basename(file))
        success = False
        try:
            if args.arm:
                success = arm_pe_file(file, altered_file, sha_value_dict)
            else:
                success = disarm_pe_file(file, altered_file, record_file)

        except Exception as e:
            logger.warning(f"Exception:Failed to process file: {file}")
            logger.warning(f'{traceback.format_exc()}')
        if not success:
            logger.error(f"Failed to process file: {file}")
            logger.error(f'{traceback.format_exc()}')


if __name__ == "__main__":
    t1 = timer()
    args = get_args()
    main(args)
    t2 = timer()
    logger.info(f'tik tok: {t2 - t1:.2f} seconds')


# python PE_modifier.py --filelist example_disarm.txt --altered_folder altered

# python PE_modifier.py --filelist example_arm.txt --altered_folder armed --arm