from argparse import ArgumentParser
import glob
from multiprocessing import Process, Pool, Queue
import numpy as np
import os
import pandas as pd
import time

# process an individual file, return a tuple with file name and mean value of the specified channel
def processing_file(file_name, channel_id):
    df = pd.read_csv(file_name, sep=";")

    col = f"c{channel_id}"
    mean_val = df[col].mean()

    return (os.path.basename(file_name), mean_val)

# run processing with a list of files and a specific channel and put results in the given queue
def processing_multi_files(my_files, channel_id, queue):
    results = []    
    for file_name in my_files:
        single_file_result = processing_file(file_name, channel_id)
        results.append(single_file_result)

    queue.put(results)

# run multiprocessing using multiprocessing.Queue
def run_mutiprocesing_queue(num_workers, file_partitions, channel_id):
    processes = []
    queues = []

    for idx in range(num_workers):
        q = Queue()

        # get the files for this worker
        my_files = file_partitions[idx]

        # TODO: create a new process for processing these files using the processing_multi_files function above
        p = Process(target=custom_processing_function, args=(arg1, arg2, q))

        p.start()
        processes.append(p)
        queues.append(q)

    for p in processes:
        p.join()

    # get the result from the queues: brute-force tallying across the queues
    combined_results = []
    for q in queues:
        result = q.get()

        # Append res to combined results
        for res in result:
            combined_results.append(res)

    return combined_results

# run multiprocessing using multiprocessing.Pool
def run_mutiprocesing_pool(num_workers, files, channel_id):
    results = []

    # TODO: Create a pool of workers and map the processing_file function to processing each file
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(custom_processing_function, [(f, channel_id) for f in files])

    return results


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
    
    # using multiprocessing.Queue
    num_files_per_worker = len(files) // num_workers
    partitions = [files[i:i + num_files_per_worker] for i in range(0, len(files), num_files_per_worker)]
    results = run_mutiprocesing_queue(num_workers, partitions, channel_id)

    # using multiprocessing.Pool
    #results = run_mutiprocesing_pool(num_workers, files, channel_id)

    end_time = time.time()
    print('Elapsed time (seconds): ', end_time - start_time)

    # Print results
    for res in results:
       print(f"Mean of channel {channel_id} in file {res[0]}: {res[1]}")
