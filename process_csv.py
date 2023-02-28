#!/usr/bin/env python3

"""Pre-processing code for HEPMASS"""

import os
import argparse
import numpy as np
import pandas as pd


def process(src: str, dst: str):
    # load the csv and cast to `np.float32`
    print(f'> reading csv at "{src}"...')
    df = pd.read_csv(src, dtype=np.float32, na_filter=None)

    print('> start processing...')
    # rename the label column to "type" was "# label"
    df.rename(columns={df.columns[0]: 'type'}, inplace=True)

    # replace the mass value "499.99" with "500"
    mass = np.sort(df['mass'].unique())
    df.loc[df['mass'] == mass[0], 'mass'] = 500.0

    # save the new csv
    print(f'> saving the new csv at "{dst}"')
    df.to_csv(dst, index=False)

    print('> processing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='source path of csv file')
    parser.add_argument('dst', help='destination path of where to save the processed csv')
    parser.add_argument('-d', '--delete', help='whether or not to delete the source csv (default: False)',
                        action='store_true')

    args = parser.parse_args()

    process(src=args.src, dst=args.dst)

    if args.delete:
        os.remove(args.src)
        print(f'> deleted source csv at "{args.src}"')
