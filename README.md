# distributed-data-processing
This repo contains the scripts for the Distributed data processing with Python workshop.

After logging in to Midway3, activate the environment

```
module load python/miniforge-25.3.0
source activate ddp
```

To create the data files for processing, run
```
python3 generate_data_files.py
```

or 
```
python3 generate_data_files.py --num-files=10000 --num-rows=32000 --num-channels=128
```

to create 10000 CSV files, each with 32000 rows and 128 channels (columns).

The scripts in this repo are as follows:

* `processing_serial.py`: serial processing, baseline
* `processing_multi.py`: parallel processing with multiprocessing on a single node
* `processing_pyspark.py`: parallel processing with PySpark on a single node
* `processing_dask.py`: parallel processing with Dask on a single node
* `processing_multi_nodes.py`: parallel processing with mpi4py across multiple nodes
* `processing_hdf5.py`: parallel processing a big HDF5 file


The 2 scripts `processing_multi.py` and `processing_multi_nodes.py` need to be filled in by the attendees to run.



