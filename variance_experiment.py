from mean_embedding import MeanEmbedding
from var_dataset import VarianceDifferenceProvider
import time
import gc
import sys
import numpy

__author__ = 'kcx'


from numpy import  mean


REPETITIONS = 100

DIMENSION = [5,100,200,300,400,500]
SAMPLE_SIZE = 10000


def reject_and_time(test, alpha):
    t1 = time.time()
    pvalue = test.compute_pvalue()
    t2 = time.time()
    return [pvalue < alpha, (t2 - t1)]

def simulation(num_freq, generator, alpha):
    test = MeanEmbedding(generator, scale=2.0 ** (-8), number_of_frequencies=num_freq)
    rejection_rates = reject_and_time(test, alpha)

    gc.collect()
    return rejection_rates


def simulations(num_freq, data_generator,dim, alpha=0.01):
    join_samples = numpy.array([simulation(num_freq, data_generator, alpha) for _ in range(REPETITIONS)])
    samples = mean(join_samples, 0)
    print('dimension: ', dim, ' power: ', samples[0])
    sys.stdout.flush()
    return samples


def run_H1_simulations():
    number_random_frequencies = 3
    return [simulations(number_random_frequencies, VarianceDifferenceProvider(SAMPLE_SIZE, dim,sd=numpy.sqrt(2.0)),dim) for dim in DIMENSION]


def run_H0_simulations():
    number_random_frequencies = 3
    return [simulations(number_random_frequencies, VarianceDifferenceProvider(SAMPLE_SIZE, dim, sd=1),dim) for dim in DIMENSION]



if __name__ == "__main__":
    t1 = time.time()
    print('===H1  ')
    run_H1_simulations()
    print('===H1  ')
    run_H0_simulations()