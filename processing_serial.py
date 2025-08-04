from argparse import ArgumentParser
import glob
import numpy as np
import pandas as pd
import os
import time

# process an individual file
def processing_file(file_name, channel_id):
    df = pd.read_csv(file_name, sep=";")

    col = f"c{channel_id}"
    mean_val = df[col].mean()

    return (os.path.basename(file_name), mean_val)

if __name__ == '__main__':
    
    data_folder = "./"
    pattern = "data_*.csv"
    verbose = False
    channel_id = 1

    parser = ArgumentParser()
    parser.add_argument("-p", "--pattern", dest="pattern", default=pattern, help="File name pattern")
    parser.add_argument("-d", "--data-folder", dest="data_folder", default=data_folder, help="Data folder")
    parser.add_argument("-c", "--channel-id", dest="channel_id", default="", help="channel")
    parser.add_argument("-v", "--verbose",   dest="verbose",   default=False, action='store_true', help="Verbose output")

    args = parser.parse_args()
    pattern = args.pattern
    verbose = args.verbose
    if len(args.channel_id) > 0:
        channel_id = int(args.channel_id)

    pattern = os.path.join(data_folder, "data_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    start_time = time.time()

    for file_name in files:

        single_file_result = processing_file(file_name, channel_id)
        print(f"Mean of channel {channel_id} in {file_name}: {single_file_result[1]}")


    end_time = time.time()
    print('Elapsed time (seconds): ', end_time - start_time)