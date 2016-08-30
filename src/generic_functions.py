import operator
import numpy as np


def add_dicts(d1, d2):
    return {sa: d1[sa] + d2[sa] for sa in d1}

def dict_argmax(dd):
    return max(dd.iteritems(), key=operator.itemgetter(1))[0]

def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)

def sample_gaussian(loc, scale, shape):
    if scale == 0:
        return loc * np.ones(shape)
    else:
        return np.random.normal(loc, scale, shape)

def update_gaussian_posterior_mean(prior, observations, tau=1):
    mu0, tau0 = prior
    tau1 = tau0 + tau * len(observations)
    mu1 = (mu0 * tau0 + sum(observations) * tau) / tau1
    return mu1
