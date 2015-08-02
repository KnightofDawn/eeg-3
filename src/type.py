import os
import numpy as np

class EegData:
  TYPE_DATA = 'data'
  TYPE_EVENTS = 'events'

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
    else:
      print 'Error: unknown eeg_type when saving'
      return

    with open(output_path, 'wb') as output:
      np.save(output, self.data)

  def load(self, input_dir, input_filename):
    # Note that we need to specify the file name
    input_path = os.path.join(input_dir, input_filename)

    r = input_filename.rfind('.bin')
    input_filename = input_filename[0 : r]

    r = input_filename.rfind('_')
    self.name = input_filename[0 : r]

    if input_filename[r + 1 : ] == 'data':
      self.type = self.TYPE_DATA
    elif input_filename[r + 1 : ] == 'events':
      self.type = self.TYPE_EVENTS
    else:
      print 'Error: unkonwn eeg_type when loading'
      return

    self.data = np.load(input_path)

  def to_string(self):
    result = ''
    for i, row in enumerate(self.data):
      result += '{}_{},{}\n'.format(self.name, i, self.row_to_string(row))
    return result

  @staticmethod
  def row_to_string(row):
    return ','.join([str(x) for x in row])


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
