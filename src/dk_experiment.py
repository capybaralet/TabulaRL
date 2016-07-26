import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents

seed = np.random.randint(10000) #1
numpy_rng  = np.random.RandomState(seed)

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

# PSRL / ENV params
alg = finite_tabular_agents.PSRL
nEps=100
epLen = 50
scaling = 1
query_function = query_functions.QueryFirstNVisits(env, agent, query_cost, 4)

targetPath = 'test.csv'

# hyper-parameters:
grid_width = 10
query_cost = .01
prob_zero_reward = .9

# states (lexical order)
states = range(grid_width**2)

# reward probabilities
reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))
print reward_probabilities

# Make the environment
env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)

f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

# Make the agent
agent = alg(env.nState, env.nAction, env.epLen,
                          scaling=scaling, 
                          P_true=env.P, R_true=False)

# Run the experiment
run_finite_tabular_experiment(agent, env, f_ext, nEps, seed,
                    recFreq=1000, fileFreq=10000, targetPath=targetPath, query_function=query_function)
