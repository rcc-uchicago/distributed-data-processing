from argparse import ArgumentParser
import glob
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import os
import time

# process an individual file
def processing_file(file_url, channel_id):
    df = pd.read_csv(file_url, sep=";")

    col = f"c{channel_id}"
    mean_val = df[col].mean()

    return (os.path.basename(file_url), mean_val)

if __name__ == '__main__':

    data_folder = "./"
    pattern = "data_*.csv"
    verbose = False
    channel_id = 1
    num_workers = 2
    num_cores_per_worker = 1

    parser = ArgumentParser()
    parser.add_argument("-p", "--pattern", dest="pattern", default=pattern, help="File name pattern")
    parser.add_argument("-d", "--data-folder", dest="data_folder", default=data_folder, help="Data folder")
    parser.add_argument("-c", "--channel-id", dest="channel_id", default="", help="channel")
    parser.add_argument("-v", "--verbose",   dest="verbose",   default=False, action='store_true', help="Verbose output")
    parser.add_argument("-n", "--num-workers", dest="num_workers", default="", help="number of workers")
    parser.add_argument("-w", "--num-cores-per-worker", dest="num_cores_per_worker", default="", help="number of workers")

    args = parser.parse_args()

    pattern = args.pattern
    verbose = args.verbose
    if len(args.data_folder) > 0:
        folder = args.data_folder
    else:
        raise FileNotFoundError(f"Requires a valid data folder")

    if len(args.channel_id) > 0:
        channel_id = int(args.channel_id)

    if len(args.num_workers) > 0:
        num_workers = int(args.num_workers)

    if len(args.num_cores_per_worker) > 0:
        num_cores_per_worker = int(args.num_cores_per_worker)

    pattern = os.path.join(folder, "data_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    # create Spark session configured for parallel execution
    spark = SparkSession.builder \
        .appName("ParallelBigDataProcessing")    \
        .master("local[*]")    \
        .config("spark.executor.instances", f"{num_workers}") \
        .config("spark.executor.cores", f"{num_cores_per_worker}")     \
        .config("spark.executor.memory", "1g")   \
        .getOrCreate()

    # create the list of files
    file_urls = [f"file://{os.path.abspath(f)}" for f in files]

    # set up Spark parallelization: numSlices = number of partitions Spark split the list of files
    #   greater than num_workers (executors) t
    rdd = spark.sparkContext.parallelize(file_urls, numSlices=2*num_workers)

    start_time = time.time()

    results = rdd.map(lambda f: processing_file(f, channel_id)).collect()

    end_time = time.time()
    print('Elapsed time (seconds): ', end_time - start_time)

    for file_name, value in results:
        print(f"{file_name}: {value}")

    spark.stop()
