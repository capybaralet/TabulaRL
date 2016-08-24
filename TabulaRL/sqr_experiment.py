import numpy as np
import argparse
import gridworld
import finite_tabular_agents
from query_functions import QueryFixedFunction

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
import pandas 

from sqr import *
from agent_sim import performance_rollouts

seed = 2 
numpy_rng = np.random.RandomState(seed)

#ENV
grid_width=4
epLen = 2 * grid_width - 1 + 8
num_episodes = 53
reward_sd = 2

env = gridworld.make_gridworld(grid_width, epLen, {}, reward_sd)
env.num_episodes = num_episodes


def randomWorld():
    rewards = np.zeros(env.nState)

    col = np.random.randint(grid_width)
    state = (grid_width - col - 1)*grid_width + col
    rewards[state] = 1

    rewards = gridworld.reward_for_action(rewards, action=0)

    world = gridworld.make_gridworld(grid_width, epLen, rewards, reward_sd)

    world.num_episodes = num_episodes
    return world

# AGENT
query_cost=1.5
reward_tau = reward_sd**-2

ns = [0,1,2,4,8]
qfunctions = [ QueryFixedFunction(query_cost, lambda s,a: (a==0)*n) for n in ns ]



prior = fillPrior(env, { (s,0) : (0, 1) for s in range(env.nState) }, (0, 10e10))

agents = [ 
        MakeSQRAgent(reward_tau, env, prior, qfunctions, iters=20),
        MakePSRLLimitedQueryAgent(reward_tau, env, prior, QueryFixedFunction(query_cost, lambda s,a: (a==0)*1)),
        MakePSRLLimitedQueryAgent(reward_tau, env, prior, QueryFixedFunction(query_cost, lambda s,a: (a==0)*0))
        ]
labels = ['sqr', 'limit query n=1','limit query n=0' ]
worlds = [ randomWorld() for i in range(50) ]


data = performance_rollouts(worlds, agents, labels)
data = pandas.DataFrame(data, columns='n,R,cumReward,cumQueryCost,perf,cumRegret,agent'.split(','))
data.to_csv('sqr_experiment.csv')
print "n chosen", agents[0].query_function.func(0,0)

