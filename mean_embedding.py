from warnings import warn

from scipy.stats import chi2

from numpy import mean, transpose, cov, shape
import numpy
from numpy.linalg import solve, LinAlgError


def mahalanobis_distance(difference, num_random_features):
    num_samples, _ = shape(difference)
    sigma = cov(transpose(difference))

    try:
        numpy.linalg.inv(sigma)
    except LinAlgError:
        warn('covariance matrix is singular. Pvalue returned is 1.1')
        return 1.1
    mu = mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(solve(sigma, transpose(mu)))

    return chi2.sf(stat, num_random_features)


class MeanEmbedding():

    def __init__(self, data_provider, scale=1, number_of_frequencies=5):
        self.data_x, self.data_y = data_provider.get_data()
        self.num_samples = data_provider.get_num_samples()
        self.data_x *= scale
        self.data_y *= scale
        self.number_of_frequencies = number_of_frequencies
        self.scale = scale

    def get_estimate(self, data, point):
        z = data - self.scale * point
        z2 = numpy.linalg.norm(z, axis=1) ** 2
        return numpy.exp(-z2 / 2.0)


    def get_difference(self, point):
        return self.get_estimate(self.data_x, point) - self.get_estimate(self.data_y, point)


    def vector_of_differences(self, dim):
        points = numpy.random.randn(self.number_of_frequencies, dim)
        a = [self.get_difference(point) for point in points]
        return numpy.array(a).T

    def compute_pvalue(self):
        _, dimension = numpy.shape(self.data_x)
        obs = self.vector_of_differences(dimension)

        return mahalanobis_distance(obs, self.number_of_frequencies)

