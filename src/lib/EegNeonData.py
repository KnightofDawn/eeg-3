import os
from neon.datasets.dataset import Dataset
import numpy as np
import pandas as pd

class EegNeonData(Dataset):
    '''
    Load the EEG data. In order to conserve memory, the minibatches are
    constructed on an on-demand basis.
    An instance of this class is created for each subject.
    '''
    def __init__(self, **kwargs):
        self.nchannels = 32
        self.nclasses = 6
        self.__dict__.update(kwargs)
        self.loaded = False
        self.mean = None

    def setwin(self, **kwargs):
        self.__dict__.update(kwargs)
        assert self.winsize % self.subsample == 0
        # This many samples to be collected for a single obesrvation.
        # The samples are picked by subsampling over a window, the size
        # of which is specified by winsize.
        self.nsamples = self.winsize // self.subsample

    def readfile(self, path, data, inds=None):
        df = pd.read_csv(path, index_col=0)
        filedata = np.float32(df.values)
        data = filedata if data is None else np.vstack((data, filedata))
        # Indices are saved to generate the submission file.
        inds = df.index if inds is None else np.hstack((inds, df.index))
        return data, inds

    def readfiles(self, data_category, serlist):
        '''
        Read the serieses specified by argument
        '''

        data_dir = os.path.join(self.data_dir, data_category)
        data = labs = inds = None

        for series in serlist:
            filename = 'subj{}_series{}_data.csv'.format(self.subj, series)
            filepath = os.path.join(data_dir, filename)
            data, inds = self.readfile(filepath, data, inds)

            if data_category == 'train':
                filename = filename.replace('data', 'events')
                filepath = os.path.join(data_dir, filename)
                labs, _ = self.readfile(filepath, labs)
            else:
                nrows = data.shape[0]
                labs = np.zeros((nrows, self.nclasses), dtype=np.float32)

        return data, labs, inds

    def prep(self, data):
        # TODO: Add your pre-processing code here
        if self.mean is None:
            self.mean = data.mean()
            self.std = data.std()

        data -= self.mean
        data /= self.std
        return data

    def load(self, **kwargs):
        if self.loaded:
            return
        self.__dict__.update(kwargs)

        if validate:
            train, train_labs, _ = self.readfiles('train', [7])
            test, test_labs, self.testinds = self.readfiles('train', [8])
        else:
            train, train_labs, _ = self.readfiles(self.data_dir, 'train', range(1, 9))
            test, test_labs, self.testinds = self.readfiles('test', [9, 10])

        self.inputs['train'] = self.prep(train)
        self.targets['train'] = train_labs
        self.inputs['test'] = self.prep(test)
        self.targets['test'] = test_labs
        self.loaded = True

    def init_mini_batch_producer(self, batch_size, setname, predict):
        '''
        This is called by neon once before training and then to switch from
        training to inference mode.
        '''
        self.setname = setname

        # Number of elements in a single observation.
        obsize = self.nchannels * self.nsamples

        self.batchdata = np.empty((obsize, self.batch_size))
        self.batchtargets = np.empty((self.nclasses, self.batch_size))
        nrows = self.inputs[setname].shape[0]

        # We cannot use the first (winsize - 1) targets because there isn't
        # enough data before their occurence
        nbatches = (nrows - self.winsize + 1) // self.batch_size

        # This variable contains a mapping to pick the right target given
        # a zero-based index.
        self.inds = np.arange(nbatches * self.batch_size) + self.winsize - 1

        if predict is False:
            # Shuffle the map of indices if we are training
            np.random.seed(0)
            np.random.shuffle(self.inds)

        return nbatches

    def get_mini_batch(self, batch):
        '''
        Called by neon when it needs the next minibatch.
        '''
        inputs = self.inputs[self.setname]
        targets = self.targets[self.setname]
        lag = self.winsize - self.subsample
        base = batch * self.batch_size

        for col in range(self.batch_size):

            # Use the saved mapping to retrieve the correct target
            end = self.inds[base + col]
            self.batchtargets[:, col] = targets[end]

            # We back up from the index of the target and sample over
            # the defined window to construct an entire observation.
            rowdata = inputs[end-lag:end+1:self.subsample]

            # Transpose to make the data from each channel contiguous.
            self.batchdata[:, col] = rowdata.T.ravel()

        # Copy to the accelerator device (in case this is running on a GPU).
        devdata = self.backend.empty(self.batchdata.shape)
        devtargets = self.backend.empty(self.batchtargets.shape)
        devdata[:] = self.batchdata
        devtargets[:] = self.batchtargets
        return devdata, devtargets






if __name__ == '__main__':
    eeg_ds = EegNeonData(subj=1)

    ROOT_DIR = '../../'
    data_dir = os.path.join(ROOT_DIR, 'data')


    data, labs, inds = eeg_ds.readfiles(data_dir, 'train', [3])
    print(data.shape)
    print(labs.shape)
    print(inds.shape)



