import sys
import logging
import numpy as np
import pandas as pd
from neon.backends import gen_backend
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score as auc

sys.path.append('../lib')
from EegNeonData import EegNeonData
from ConvNetNeon import ConvNet


logging.basicConfig(level=30)
logger = logging.getLogger()


# Global variables
validate = True
data_dir = '../../data'


def run(subj):
    """
    Train and perform inference on data from a single subject.
    """
    try:
        backend = gen_backend(rng_seed=0, gpu='nervanagpu')
    except:
        backend = gen_backend(rng_seed=0)

    ds = EegNeonData(subj=subj, validate=validate, data_dir=data_dir)
    sumpreds = None
    winlist = [1024] if validate else [768, 1024, 1280, 1536]
    for winsize in winlist:
        ds.setwin(winsize=winsize, subsample=16)
        network = ConvNet(backend, ds, subj)
        labs, preds, inds = network.fit().predict()
        if sumpreds is None:
            sumpreds = preds
        else:
            sumpreds += preds
    if validate:
        aucs = [auc(labs[:, i], sumpreds[:, i]) for i in range(ds.nclasses)]
        print('Subject %d AUC %.4f' % (subj, np.mean(aucs)))
    return labs, sumpreds, inds


if __name__ == '__main__':
    print('\'validate\' is %s' % validate)

    # Launch a separate process for each subject
    nsubjects = 1
    pool = Pool()
    results = pool.map(run, range(1, nsubjects + 1))
    pool.close()

    labs = np.vstack([tup[0] for tup in results])
    preds = np.vstack([tup[1] for tup in results])

    if validate:
        # Compute AUC metric.
        nclasses = labs.shape[1]
        aucs = [auc(labs[:, i], preds[:, i]) for i in range(nclasses)]
        print('Mean AUC %.4f' % np.mean(aucs))
    else:
        # Generate submission file.
        columns = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
                   'LiftOff', 'Replace', 'BothReleased']
        inds = np.hstack([tup[2] for tup in results])
        subm = pd.DataFrame(index=inds, columns=columns, data=preds)
        subm.to_csv('subm.csv', index_label='id', float_format='%.4f')

    print('Done.')