"""
WIP

Comparing stochastic termination to discounting in Monte Carlo control
"""

import numpy
np = numpy
import numpy.random

from pylab import *

import time

from doina_class.algorithms import iterative_policy_evaluation, EVMC
from doina_class.environments import env_dict
from doina_class.policies import EpsilonGreedy
from doina_class.utilities import err_plot, sample


# TODO: actually compute complexity? (nsteps is not varying as much as it should, I think...)
# TODO: read the paper or something...

#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='random_walk')
parser.add_argument('--gamma', type=int, default=.95)
parser.add_argument('--size', type=int, default=19)
#parser.add_argument('--lr', type=float, default=.4) # learning rate
#parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--num_trials', type=int, default=30)
parser.add_argument('--points_per_trial', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--time_scale', type=float, default=.001)
#parser.add_argument('--stochastic_termination', type=float, default=0)
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)

if seed is None:
    seed = np.random.randint(2**32 - 1)


#----------------------------------------
# RUN 

RMSs = np.zeros((5, num_trials, points_per_trial))

for nn, stochastic_termination in enumerate([0, .25, .5, .75, 1.]):

    env = env_dict[environment](size)
    env.gamma = gamma
    mu = np.ones((env.nS, env.nA)) * 1./env.nA

    assert eps == 1
    Q_pi = iterative_policy_evaluation(mu, env, return_Q=1)

    for trial in range(num_trials):
        Q = np.zeros((env.nS, env.nA))
        C = np.zeros((env.nS, env.nA))
        t0 = time.time()
        num_episodes = 0
        for step in range(points_per_trial):
            while time.time() < t0 + time_scale * step:
                num_episodes += 1
                C, Q = EVMC(env, pi=mu, mu=mu, num_episodes=1, C=C, Q=Q, stochastic_termination=stochastic_termination)
            RMSs[nn, trial, step] = (np.mean((Q - Q_pi)**2))**.5
        print num_episodes

    np.save("stochastic_termination__RMSs.npy", RMSs)


    if 1: # plot
        figure(33)
        err_plot(RMSs[nn], label="stochastic_termination=" + str(stochastic_termination))
        legend()
        xlabel('milliseconds')
        ylabel('RMS of Q')
        show()





