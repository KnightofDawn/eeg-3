from __future__ import print_function
import os

import numpy as np


class EegData:
    TYPE_EVENTS = 'events'
    TYPE_DATA = 'data'
    TYPE_FFT0 = 'fft0'
    TYPES = [TYPE_DATA, TYPE_EVENTS]

    def __init__(self, name=None, eeg_type=None, data=None):
        self.name = name
        self.type = eeg_type
        self.data = data

    def save(self, output_dir):
        # Note that the file name should be generated from self.name
        # Therefore the input parameter is output_dir
        if self.type == self.TYPE_DATA:
            output_path = os.path.join(output_dir, self.name + '_data.bin')
        elif self.type == self.TYPE_EVENTS:
            output_path = os.path.join(output_dir, self.name + '_events.bin')
        elif self.type == self.TYPE_FFT0:
            output_path = os.path.join(output_dir, self.name + '_fft0.bin')
        else:
            print('Error: unknown eeg_type when saving')
            return

        with open(output_path, 'wb') as output:
            np.save(output, self.data)

    def load(self, input_dir, input_filename):
        # Note that we need to specify the file name
        input_path = os.path.join(input_dir, input_filename)

        r = input_filename.rfind('.bin')
        input_filename = input_filename[0: r]

        r = input_filename.rfind('_')
        self.name = input_filename[0: r]

        if input_filename[r + 1:] == 'data':
            self.type = self.TYPE_DATA
        elif input_filename[r + 1:] == 'events':
            self.type = self.TYPE_EVENTS
        else:
            print('Error: unknown eeg_type when loading')
            return

        self.data = np.load(input_path)

    def __str__(self):
        return ''.join(
            map(
                lambda i_and_row: '{}_{},{}\n'.format(
                    self.name, i_and_row[0], self.row_to_string(i_and_row[1])
                ),
                enumerate(self.data),
            )
        )

    @staticmethod
    def row_to_string(row):
        return ','.join([str(x) for x in row])

def from_load(input_dir, input_filename):
    data = EegData()
    data.load(input_dir, input_filename)
    return data

def output_submission(prediction_array, output_path):
    for prediction in prediction_array:
        if prediction.type != EegData.TYPE_EVENTS:
            print('Error: prediction_array contains a invalid prediction')
            return

    prediction_array.sort()

    header = 'id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n'

    buf_arr = map(lambda x: str(x), prediction_array)
    content = header + ''.join(buf_arr)

    with open(output_path, 'w') as output_file:
        output_file.write(content)


def extract_type(file_name):
    l = file_name.rfind('_')
    r = file_name.rfind('.bin')
    extracted = file_name[l + 1: r]
    if extracted in EegData.TYPES:
        return extracted
    else:
        return ''


# Load all data from a folder, for a list of types
def load_folder(input_dir, types):
    if not os.path.isdir(input_dir):
        print('Error: invalid input_dir "{}"'.format(input_dir))

    input_files = os.listdir(input_dir)

    if not type(types) is list:
        types = [types]

    input_files = filter(lambda x: extract_type(x) in types, input_files)
    data_arr = map(lambda x: EegData.from_load(input_dir, x), input_files)
    return data_arr

def load_folder_as_dict(input_dir, types):
    l = load_folder(input_dir, types)
    return dict(zip([x.name for x in l], l))

# Test examples

# v = EegData()
# v.load('e:/eeg/data/random', 'subj1_series1_events.bin')
# v.save('e:/eeg/data/randomr/')
# print v.name, v.type, v.data.shape

# v = EegData()
# v.name = 'test'
# v.type = 'test'
# v.data = np.arange(6).reshape(2, 3)
# print v.data
# print v.to_string()

# v = EegData()
# v.name = 'test1'
# v.data = np.arange(12).reshape(2, 6)
# v.type = EegData.TYPE_EVENTS;

# u = EegData()
# u.name = 'test0'
# u.data = np.arange(12).reshape(2, 6)
# u.type = EegData.TYPE_EVENTS;

# l = [u, v]
# EegData.output_submission(l, 'test.csv')

# v = EegData.from_load('e:/eeg/data/random', 'subj1_series1_events.bin')
# print v.name, v.data.shape

# load_folder('e:/eeg/data/train_bin', EegData.TYPE_DATA)
