from __future__ import print_function
from builtins import range
import os
import sys

import numpy as np

def parse_csv(file_name, ncol):
    with open(file_name) as text:
        data = np.loadtxt(
            text,
            dtype = int,
            delimiter=',',
            skiprows=1,
            usecols = range(1, ncol + 1)
        )
        return data

def save_binary(file_name, arr):
    with open(file_name, 'wb') as output:
        np.save(output, arr)

def load_binary(file_name):
    return np.load(file_name)

def main():
    if len(sys.argv) < 3:
        print('Usage: python input2binary.py input_dir/ output_dir/ [--test]')
        return

    test_flag = False
    if len(sys.argv) >= 4:
        if sys.argv[3] == '--test':
            test_flag = True

    _, input_dir, output_dir = sys.argv[0 : 3]
    input_dir = input_dir.strip()
    output_dir = output_dir.strip()

    if not os.path.isdir(input_dir):
        print('Error: invalid input_dir "{}"'.format(input_dir))
        return

    if not os.path.isdir(output_dir):
        print('Error: invalid output_dir "{}"'.format(output_dir))
        return

    input_files = os.listdir(input_dir)

    if len(input_files) == 0:
        return

    for input_file in input_files:
        r = input_file.rfind('.csv')
        if r == -1:
            continue

        output_file = input_file[0 : r] + '.bin'

        input_full = os.path.join(input_dir, input_file)

        if input_file.rfind('_data.csv') != -1:
            # data file
            data = parse_csv(input_full, 32)
        else:
            data = parse_csv(input_full, 6)

        output_full = os.path.join(output_dir, output_file)
        save_binary(output_full, data)

        print('{} => {}, {} rows, {} cols'.format(
            input_file, output_file, data.shape[0], data.shape[1]
        ))

        if test_flag:
            readback = load_binary(output_full)
            if np.all(data == readback):
                print('test pass!')


if __name__ == '__main__':
    main()
