from argparse import ArgumentParser
import numpy as np

if __name__ == '__main__':
    num_files = 20
    num_rows = 1000
    num_channels = 4
    dt = 0.01

    parser = ArgumentParser()
    parser.add_argument("-s", "--num-files", dest="num_files", default="", help="Number of files")
    parser.add_argument("-r", "--num-rows", dest="num_rows", default="", help="Number of rows")
    parser.add_argument("-c", "--num-channels", dest="num_channels", default="", help="Number of channels")

    args = parser.parse_args()
    if len(args.num_files) > 0:
        num_files = int(args.num_files)
    if len(args.num_rows) > 0:
        num_rows = int(args.num_rows)
    if len(args.num_channels) > 0:
        num_channels = int(args.num_channels)

    for i in range(num_files):
        with open(f"data_{i}.csv", "w") as f:
            for n in range(num_rows):
                header = f"#idx"
                for m in range(num_channels):
                    header += f";c{m}"
                header += "\n"
            f.write(header)

            for n in range(num_rows):
                s = f"{n}"
                for m in range(num_channels):
                    freq = 2*np.pi*i
                    value = 4*i*np.sin(freq*n*dt) + 6.0*i*np.cos(-2.0*freq*m*dt)
                    s += f";{value}"
                s += "\n"    
                f.write(s)

    print(f"Generated {num_files} files each with {num_channels} channels, each with {num_rows} rows")