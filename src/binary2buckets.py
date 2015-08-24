import os
import sys
import math
import numpy as np
import multiprocessing as mp
from functools import partial
from lib import EegData
from lib import DataPipe

def bin2bucket_batch(raw_data):
    fft_data_list = map(bin2bucket, raw_data.values())
    return dict(zip([x.name for x in fft_data_list], fft_data_list))

def bin2bucket(eeg_data):
    if eeg_data.type != EegData.EegData.TYPE_DATA:
        raise Exception('Error', 'input is not TYPE_DATA')
    return EegData.EegData(
        name = eeg_data.name,
        eeg_type = EegData.EegData.TYPE_FFT0,
        data = bin2bucket_impl(eeg_data.data)
    )

def bin2bucket_impl(data):
    # Bands
    bucket = [0, 2, 4, 8, 16, 32, 64]

    # 500 Hz, 512 points -> 500/512 * k Hz for X[k]
    sample_rate = 500.0
    n_points = 512

    base_frequency = sample_rate / n_points

    class FFTBucket(DataPipe.DataPipeFunction):
        input_depth = 512
        input_width = 1
        output_width = len(bucket) - 1
        def calc(self, input):
            fft_data = np.abs(np.fft.rfft(input, axis=0)) / (self.input_depth / 2)
            fft_data[0] /= 2
            bucket_data = np.zeros([1, self.output_width])
            j = 0
            for i in xrange(self.output_width):
                while j * base_frequency < bucket[i + 1] and j < self.input_depth:
                    bucket_data[0, i] += fft_data[j] * fft_data[j]
                    j += 1
            return np.sqrt(bucket_data)

    pipe = DataPipe.DataPipeOffline(
        depth = n_points,
        default = 0,
        workers = 1
    )

    pipe.set_function(FFTBucket)
    return np.concatenate(map(pipe.calc, data.T), axis=1)

def process(input_dir, output_dir, input_filename):
    data = EegData.from_load(input_dir, input_filename)
    new_data = bin2bucket(data)
    new_data.save(output_dir)
    pass

def main():
    if len(sys.argv) < 3:
        print 'Usage: python bin2fft.py input_dir/ output_dir/'
        return

    _, input_dir, output_dir = sys.argv[0 : 3]
    input_dir = input_dir.strip()
    output_dir = output_dir.strip()

    if not os.path.isdir(input_dir):
        print 'Error: invalid input_dir "{}"'.format(input_dir)
        return

    if not os.path.isdir(output_dir):
        print 'Error: invalid output_dir "{}"'.format(output_dir)
        return

    input_files = os.listdir(input_dir)
    types = [EegData.EegData.TYPE_DATA]
    input_files = filter(lambda x: EegData.extract_type(x) in types, input_files)
    pool = mp.Pool(16)
    func = partial(process, input_dir, output_dir)
    pool.map(func, input_files)

def test():
    v = np.ones([512, 1]) + np.matrix([math.sin(2 * math.pi * x * 200 / 500) for x in range(512)]).transpose()
    print v.shape
    print bin2bucket_impl(v)

if __name__ == '__main__':
    main()
    # test()
