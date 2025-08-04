from argparse import ArgumentParser
import glob
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import os
import pandas as pd
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
    num_workers = 4

    parser = ArgumentParser()
    parser.add_argument("-p", "--pattern", dest="pattern", default=pattern, help="File name pattern")
    parser.add_argument("-d", "--data-folder", dest="data_folder", default=data_folder, help="Data folder")
    parser.add_argument("-c", "--channel-id", dest="channel_id", default="", help="channel")
    parser.add_argument("-n", "--num-workers", dest="num_workers", default="", help="number of workers")
    parser.add_argument("-v", "--verbose",   dest="verbose",   default=False, action='store_true', help="Verbose output")

    args = parser.parse_args()
    pattern = args.pattern
    data_folder = args.data_folder
    verbose = args.verbose
    if len(args.channel_id) > 0:
        channel_id = int(args.channel_id)

    if len(args.num_workers) > 0:
        num_workers = int(args.num_workers)
    
    pattern = os.path.join(data_folder, "data_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    
    start_time = time.time()


    #with MPIPoolExecutor() as executor:
    #    futures = [executor.submit(processing_file, f, channel_id) for f in files]
    #    results = [future.result() for future in futures]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    files_this_rank = files[rank::size]  # Distribute files across ranks
    results = [processing_file(f, channel_id) for f in files_this_rank]

    all_results = comm.gather(results, root=0)

    end_time = time.time()
    elasped_time = end_time - start_time

    total = comm.allreduce(elasped_time, op=MPI.SUM)
    if rank == 0:
        print('Elapsed time (seconds): ', total/size)

    # Print results
    for res in results:
       print(f"Mean of channel {channel_id} in file {res[0]}: {res[1]}")

