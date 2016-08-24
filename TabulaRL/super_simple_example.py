import numpy as np
import gridworld
import finite_tabular_agents
from query_functions import QueryFirstNVisits

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
import pandas 

from sqr import *

seed = 2 
numpy_rng = np.random.RandomState(seed)

#ENV
grid_width=4
epLen = 2 * grid_width - 1 + 8
num_episodes = 53
reward_sd = 2

env = gridworld.make_gridworld(grid_width, epLen, rewards={ (0, 0) : 1}, reward_noise=reward_sd)

# AGENT
query_cost=1.5
reward_tau = reward_sd**-2

agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
          scaling=.1, 
          P_true=env.P, R_true=None, 
          query_function=QueryFirstNVisits(query_cost, 5), 
          tau=reward_tau)

agent.R_prior = fillPrior(env, { (s,0) : (0, 1) for s in range(env.nState) }, (0, 10e10))

f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

# returns: cumReward, cumQueryCost, perf, cumRegret
results = run_finite_tabular_experiment(agent, env, f_ext, num_episodes, seed,
                    recFreq=1000, fileFreq=10000, targetPath='')   
print results
