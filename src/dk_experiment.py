'''
Script to run tabular experiments in batch mode.

author: iosband@stanford.edu
'''

import numpy
import numpy as np
import pandas as pd
import argparse
import sys
import query_functions

seed = 1#np.random.randint(10000) #1
numpy_rng  = numpy.random.RandomState(seed)

import environment
import finite_tabular_agents

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
from environment import TabularMDP

# PSRL / ENV params
nEps=1000
epLen = 50
gap = .1
#alg = finite_tabular_agents.EpsilonGreedy #PSRL
alg = finite_tabular_agents.PSRL
scaling = 1


targetPath = ('test.csv')
# hyper-parameters:
grid_width = 6
prob_random_action = 0.1
prob_random_reset = 0.001
query_cost = .01
gamma = .75 # discount factor
prob_zero_reward = .9

learning_rate = .1

# states (lexical order)
states = range(grid_width**2)

# reward probabilities
reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))
print reward_probabilities


##################################

def row_and_column(state):
	return state / grid_width, state % grid_width

def state_from_row_and_column(row, column):
	state = row * grid_width + column
	assert 0 <= state and state <= len(states) 
	return state

actions = range(5) # stay, N, E, S, W

def next_state(state, action):
	row, column = row_and_column(state)
	if action == 1 and row > 0:
		return state - grid_width
	if action == 2 and column < grid_width - 1:
		return state + 1
	if action == 3 and row < grid_width - 1:
		return state + grid_width
	if action == 4 and column > 0:
		return state - 1
	else:
		return state

def make_deterministic(epLen, nState, nAction, transition, rewards):
    """
    make the environment deterministic 
        (and potentially makes the agent know that)
    """
    R_true = {}
    P_true = {}

    for s in xrange(nState):
        for a in xrange(nAction):
            R_true[s, a] = (rewards[s], 0)

            P_true[s, a] = np.zeros(nState)
            #deterministic transitions
            P_true[s, a][transition(s, a)] = 1

    env = TabularMDP(nState, nAction, epLen)
    env.R = R_true
    env.P = P_true
    env.reset()

    prior = {}

    """
    #almost deterministic prior
    for s in xrange(nState):
        for a in xrange(nAction):
            tps = np.zeros(nState)
            tps[transition(s,a)] = 10000
            prior[s, a] = tps 
    env.P_prior = prior
    print prior
    """
    return env


# Make the environment
env = make_deterministic(epLen, grid_width**2, 5, next_state, reward_probabilities)


# Make the feature extractor
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

# Make the agent
agent = alg(env.nState, env.nAction, env.epLen,
                          scaling=scaling, 
                          P_true=env.P, R_true=False)

# Run the experiment
print targetPath
query_function = query_functions.QueryFirstNVisits(env, agent, 2, 4)
run_finite_tabular_experiment(agent, env, f_ext, nEps, seed,
                    recFreq=1000, fileFreq=10000, targetPath=targetPath, query_function=query_function)

