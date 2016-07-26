'''
Script to run tabular experiments in batch mode.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
import argparse
import sys

import environment
import finite_tabular_agents

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
from environment import TabularMDP

# hyper-parameters:
grid_width = 3
prob_random_action = 0.1
prob_random_reset = 0.001
query_cost = .01
gamma = .75 # discount factor
prob_zero_reward = .9

learning_rate = .1

# states (lexical order)
states = range(grid_width**2)

# reward probabilities
reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))



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
def make_gridworld(epLen):
    nState = 
    R_true = {}
    P_true = {}

    for a in xrange(nAction):
        # Rewards are independent of action
        R_true[0, a] = (0.5, 1)
        R_true[1, a] = (1, 0)
        R_true[2, a] = (0, 0)

        # Transitions are like a bandit
        P_true[0, a] = np.array([0, pSuccess, 1 - pSuccess])
        P_true[1, a] = np.array([0, 1, 0])
        P_true[2, a] = np.array([0, 0, 1])

    # The first action is a better action though
    P_true[0, 0] = np.array([0, pSuccess + gap, 1 - (pSuccess + gap)])

    hardBanditMDP = TabularMDP(nState, nAction, epLen)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true
    hardBanditMDP.reset()

    return hardBanditMDP



nEps=1000
epLen = 10 
gap = .1
alg = finite_tabular_agents.EpsilonGreedy #PSRL
scaling = 1
seed = 1

targetPath = ('test.csv')

# Make the environment
env = environment.make_hardBanditMDP(epLen=epLen, gap=gap,
                                     nAction=2, pSuccess=0.5)

# Make the feature extractor
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

# Make the agent
agent = alg(env.nState, env.nAction, env.epLen,
                          scaling=scaling)

# Run the experiment
print targetPath
run_finite_tabular_experiment(agent, env, f_ext, nEps, seed,
                    recFreq=1000, fileFreq=10000, targetPath=targetPath)

