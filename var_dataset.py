__author__ = 'kcx'
import numpy



def multidim(num_samples, dimensions, sigma=1, mu=0):
    X = numpy.random.randn(num_samples, dimensions)
    X[:, 0] *= sigma
    X[:, 0] += mu
    return X

class VarianceDifferenceProvider():

    def get_data(self):
        return self.x, self.y

    def get_num_samples(self):
        return self.num_samples

    def __init__(self, n, dimensions, sd):
        self.num_samples = n
        self.dimension = dimensions
        self.sd = sd

    def get_data(self):
        x = multidim(self.num_samples, self.dimension, 1, 0)
        y = multidim(self.num_samples, self.dimension, self.sd, 0)
        return x, y

    def __str__(self):
        return "dimension" + str(self.dimension)