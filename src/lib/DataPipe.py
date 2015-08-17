import numpy as np
import multiprocessing as mp

class DataPipeFunction:
    """
    Atrributes:
        input_depth
        input_width
        output_width
    """
    def __init__(self):
        pass

    def calc(self, input):
        pass

class DataPipeOffline:
    def __init__(self, depth=1, default=0, workers=1):
        self.depth = depth
        self.default = default
        self.workers = workers

    def set_function(self, function):
        if function.input_depth == self.depth:
            self.function = function
        else:
            raise Exception('Error when attach function to a DataPipe')

    def calc(self, input_series):
        n = input_series.shape[0]

        if self.depth > 1:
            shape = list(input_series.shape)
            shape[0] = self.depth - 1
            default_series = np.ones(shape) * self.default
            input_series = np.concatenate((default_series, input_series))
        self.input_series = input_series

        self.function_inst = self.function()

        if self.workers == 1:
            output_list = map(self.calc, range(n))
        else:
            pool = mp.Pool(self.workers)
            output_list = map(self.calc, range(n))

        return np.concatenate(output_list)

    def calc_impl(self, begin):
        data = self.input_series[begin : (begin + self.depth), :]
        return self.function_inst.calc(data)

def test():
    v = DataPipeOffline(depth=2, default=1, workers=4)

    class myFunc(DataPipeOffline):
        input_depth = 2
        input_width = 1
        output_width = 1
        def calc(self, input):
            return (input[0] + input[1]) / 2

    v.set_function(myFunc)

    input_series = np.matrix([1, 2, 3, 4, 5, 6, 7]).transpose()
    print v.calc(input_series)
