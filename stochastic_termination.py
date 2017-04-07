"""
WIP

Comparing stochastic termination to discounting in Monte Carlo control

"""


#----------------------------------------
# TODOs

"""
TODO: is it really worse?  (it seems like it)

TODO: actually compute complexity? (nsteps is not varying as much as it should, I think...)
TODO: read the paper or something...

Why is the time so similar?
ATM, we're just doing evaluation (not control)
"""


#----------------------------------------
# imports

import numpy
np = numpy
import numpy.random

from pylab import *

import time

from doina_class.algorithms import iterative_policy_evaluation, EVMC
from doina_class.environments import env_dict
from doina_class.policies import EpsilonGreedy
from doina_class.utilities import err_plot, sample


#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='lanes')
parser.add_argument('--P_terminate', type=float, default=.001) # probability of termination for random MDP
parser.add_argument('--gamma', type=int, default=.9)
parser.add_argument('--size', type=int, default=15)
#parser.add_argument('--lr', type=float, default=.4) # learning rate
#parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--num_trials', type=int, default=30)
parser.add_argument('--points_per_trial', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--target_policy', type=str, default='mu')
parser.add_argument('--time_scale', type=float, default=.0001)
#parser.add_argument('--stochastic_termination', type=float, default=0)
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)

if seed is None:
    seed = np.random.randint(2**32 - 1)


#----------------------------------------
# RUN 

term_probs = [0,.5,1]
term_probs = [1]

RMSs = np.zeros((len(term_probs), num_trials, points_per_trial))
Ts = np.zeros((len(term_probs), num_trials, points_per_trial))

env = env_dict[environment](size, P_terminate=P_terminate)
env.gamma = gamma
mu = np.ones((env.nS, env.nA)) * 1./env.nA
assert eps == 1 # TODO: other policies
Q_pi = iterative_policy_evaluation(mu, env, return_Q=1)

if target_policy == 'mu':
    pi = mu
elif target_policy == 'control':
    pi = 'greedy'
else:
    assert False, "not implemented!"

for trial in range(num_trials):

    if environment == 'random_MDP':
        # FIXME P_terminate kwarg, ipe every time
        env = env_dict[environment](size, P_terminate=P_terminate)
        env.gamma = gamma
        mu = np.ones((env.nS, env.nA)) * 1./env.nA
        assert eps == 1 # TODO: other policies
        Q_pi = iterative_policy_evaluation(mu, env, return_Q=1)

        if target_policy == 'mu':
            pi = mu
        elif target_policy == 'control':
            pi = 'greedy'
        else:
            assert False, "not implemented!"

    for nn, stochastic_termination in enumerate(term_probs):

        Q = np.zeros((env.nS, env.nA))
        C = np.zeros((env.nS, env.nA))
        t0 = time.time()
        num_episodes = 0
        for step in range(points_per_trial):
            total_T = 0
            while time.time() < t0 + time_scale * step:
            #if 1:
                num_episodes += 1
                C, Q, T = EVMC(env, pi=mu, mu=mu, num_episodes=1, C=C, Q=Q, stochastic_termination=stochastic_termination)
                print T
                total_T += T
            RMSs[nn, trial, step] = (np.mean((Q - Q_pi)**2))**.5
            Ts[nn, trial, step] = total_T
            #print total_T
        #print num_episodes

    np.save("stochastic_termination__RMSs.npy", RMSs)


if 1: # plot
    figure()
    for nn, stochastic_termination in enumerate(term_probs):
        err_plot(RMSs[nn], label="stochastic_termination=" + str(stochastic_termination))
        legend()
        xlabel('milliseconds')
        ylabel('RMS of Q')
        show()





