

import numpy as np
import matplotlib.pyplot as plt

def onehot(x, length):
    rval = np.zeros(length)
    rval[x] = 1
    return rval

def sample(pvals):
    return np.argmax(np.random.multinomial(1, pvals))


# -----------------------------------------------------------
# visualization

def err_plot(samples, **kwargs):
    """
    plot averaged samples with (standard error) error-bars
    samples shape: (sample_n, ___)
    """
    means = samples.mean(axis=0)
    stds = samples.std(axis=0) / samples.shape[0]**.5
    plt.errorbar(range(samples.shape[1]), means, stds, **kwargs) 

