import numpy as np
from neon.models import MLP
from neon.experiments import FitExperiment as Fit
from neon.layers import FCLayer, DataLayer, CostLayer, ConvLayer, PoolingLayer
from neon.transforms import RectLin, Logistic, CrossEntropy


class ConvNet(object):
    """
    The network definition.
    """
    def __init__(self, backend, dataset, subj):
        ad = {
            'type': 'adadelta',
            'lr_params': {'rho': 0.9, 'epsilon': 0.000000001}
        }

        self.layers = []

        self.add(DataLayer(
            is_local=True,
            nofm=dataset.nchannels,
            ofmshape=[1, dataset.nsamples]))

        self.add(ConvLayer(
            nofm=64,
            fshape=[1, 3],
            activation=RectLin(),
            lrule_init=ad))

        self.add(PoolingLayer(
            op='max',
            fshape=[1, 2],
            stride=2))

        self.add(FCLayer(
            nout=128,
            activation=RectLin(),
            lrule_init=ad))

        self.add(FCLayer(
            nout=dataset.nclasses,
            activation=Logistic(),
            lrule_init=ad))

        self.add(CostLayer(
            cost=CrossEntropy()))

        self.model = MLP(num_epochs=1, batch_size=128, layers=self.layers)
        self.backend = backend
        self.dataset = dataset


    def add(self, layer):
        self.layers.append(layer)


    def fit(self):
        Fit(model=self.model, backend=self.backend, dataset=self.dataset).run()
        return self


    def predict(self):
        ds = self.dataset
        outputs, targets = self.model.predict_fullset(self.dataset, 'test')
        predshape = (ds.inputs['test'].shape[0], ds.nclasses)
        preds = np.zeros(predshape, dtype=np.float32)
        labs = np.zeros_like(preds)

        # The output returned by the network is less than the number of
        # predictions to be made. We leave the missing predictions as zeros.
        start = ds.winsize - 1
        end = start + outputs.shape[1]
        preds[start:end] = outputs.asnumpyarray().T
        labs[start:end] = targets.asnumpyarray().T
        return labs, preds, ds.testinds