from argparse import ArgumentParser
import h5py
import numpy as np
import os

def generate_hdf5_file(filename="bigdata.h5", N=1_000_000):

    # Define a structured dtype
    dtype = np.dtype([
        ("id", np.int32),
        ("value", np.float64),
        ("vector", np.float32, (3,))  # fixed-size array field
    ])
    
    # Create some dummy data
    data = np.zeros(N, dtype=dtype)
    data["id"] = np.arange(N)
    data["value"] = np.random.random(N)
    data["vector"] = np.random.random((N, 3)).astype(np.float32)

    # Write to HDF5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", data=data, chunks=True, compression="gzip")

    print(f"Created {filename} with {N} records.")

def read_segment(rank, n_procs, filename):
    with h5py.File(filename, "r") as f:
        dset = f["dataset"]
        n_total = dset.shape[0]
        chunk_size = n_total // n_procs
        start = rank * chunk_size
        end = (rank + 1) * chunk_size if rank != n_procs - 1 else n_total

        # Each process reads its segment
        data = dset[start:end]
        local_mean = data["value"].mean()
        print(f"Process {rank} read rows {start}:{end}, first id={data['id'][0]}, local mean={local_mean}")

if __name__ == "__main__":
  
    file_name = "bigdata.h5"
    num_workers = 4
    using_mpi = False

    parser = ArgumentParser()
    parser.add_argument("-p", "--file-name", dest="file_name", default=file_name, help="HDF5 file name")
    parser.add_argument("-n", "--num-workers", dest="num_workers", default="", help="number of workers")
    parser.add_argument("-m", "--using-mpi", dest="using_mpi", default=False, action='store_true', help="Using mpi4py")

    args = parser.parse_args()
    file_name = args.file_name
    if len(args.num_workers) > 0:
        num_workers = int(args.num_workers)

    using_mpi = args.using_mpi

    # Check if the file exists, if not generate it
    if not os.path.exists(file_name):
        generate_hdf5_file(file_name, N=1_000_000)

    # Start multiple processes to read segments of the HDF5 file
    if using_mpi == False:

        import multiprocessing as mp

        procs = []
        for rank in range(num_workers):
            # TODO: Create a new process for each rank and process its segment
            p = mp.Process(target=custom_processing_function, args=(rank, num_workers, file_name))
            p.start()

            procs.append(p)

        for p in procs:
            p.join()

    else:

        from mpi4py import MPI

        # Run MPI to read the HDF5 file in parallel
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Each rank opens the file independently (no driver="mpio")
        with h5py.File(file_name, "r") as f:
            my_subset = f["dataset"]
            n_total = my_subset.shape[0]

            chunk_size = n_total // size
            start = rank * chunk_size
            end = (rank + 1) * chunk_size if rank != size - 1 else n_total

            # TODO: Each rank reads only its portion
            data = ...
            local_sum = data["value"].sum()
            print(f"Rank {rank} read rows {start}:{end}, first id={data['id'][0]}, local sum={local_sum}")

            # combine results across all ranks
            global_sum = comm.allreduce(local_sum, op=MPI.SUM)

            if rank == 0:
                print(f"Global sum={global_sum}")
