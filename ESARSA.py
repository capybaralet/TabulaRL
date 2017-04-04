"""
WIP
HACKING it to see is E-SARSA is ever WORSE
"""

import numpy
np = numpy
import numpy.random

from algorithms import iterative_policy_evaluation, SARSA
from environments import env_dict
from policies import EpsilonGreedy
from utilities import sample

#----------------------------------------
# hparams

save_str = 'ESARSA'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='random_walk')
parser.add_argument('--gamma', type=int, default=1.) #
parser.add_argument('--size', type=int, default=19)
parser.add_argument('--lr', type=float, default=.4) # learning rate
parser.add_argument('--num_episodes', type=int, default=100) #
parser.add_argument('--num_trials', type=int, default=10) #
parser.add_argument('--expected', type=int, default=0) # Expected SARSA?
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)


if seed is None:
    seed = np.random.randint(2**32 - 1)
rng = numpy.random.RandomState(seed)


#----------------------------------------
# RUN 

# FIXME: doesn't work with gridworld (probably a problem in the env!)
#assert environment == 'random_walk'
env = env_dict[environment](size)
env.gamma = gamma
mu = EpsilonGreedy(eps, env)

# TODO: fix policy representations, figure out how to treat non-stationary policies
assert eps == 1
Q_pi = iterative_policy_evaluation(np.ones((env.nS, env.nA)) * 1./env.nA , env, return_Q=1)

RMS = np.zeros((num_trials, num_episodes))
for trial in range(num_trials):
    Q = np.zeros((env.nS, env.nA))
    for episode in range(num_episodes):
        Q = SARSA(mu, env, lr=lr, Q=Q, expected=expected)
        RMS[trial, episode] = (np.mean((Q - Q_pi)**2))**.5


#np.save(save_str + environment  + '_' + sigma + '___COMPLETE.npy', perfs)


if 1: # plot
    from pylab import *
    figure(33)
    if expected:
        plot(np.mean(RMS, axis=0), label="Expected SARSA")
    else:
        plot(np.mean(RMS, axis=0), label="SARSA")
    legend()
    xlabel('episode')
    ylabel('RMS of Q')
    show()





