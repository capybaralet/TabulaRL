import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents
from feature_extractor import FeatureTrueState
#from experiment import run_finite_tabular_experiment
from TabulaRL.dk_run_finite_tabular_experiment import run_finite_tabular_experiment
from environment import make_stochasticChain
#np.random.seed(1)

from ASQR_and_SQR import *

#TODO: 
"""
  Decide: 
   how many samples of R?
   how many jobs?
      1000 jobs with each setting, and run 100,000 episodes (?)
   Nmax for SQR/ASQR
  ASQR in the loop

In a pkl file, we'll log:
    visit counts
    query counts (TODO)
    desired query sets (TODO)

Things that could change our results:
    environment
    query_cost
    horizon / discounting

EXPERIMENTS:
    We run SQR/ASQR for each environment with a bunch of different sampled_R in order to get an estimate of the distribution over n_best that each gives.
        We fix the horizon/discount
        We try several different query_costs
    We run PSRL (limitedquery/not) in each environment a bunch of times to get an estimate of the performance of each method
        ^ Since horizon and cost are sort of "inversely proportional" we don't *really* need to search over both...
        ^ when we run with a fixed desired query set, a single experiment allows us to evaluate many query_costs / horizons


"""

# SETUP
# TODO: save results in a single file / database
import os
filename = os.path.basename(__file__)
save_dir = os.path.join(os.environ['HOME'], 'TabulaRL/src/results/results__' + filename)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

import datetime
timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
save_dir += '/' + timestamp 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

import argparse
parser = argparse.ArgumentParser()
# we only need different costs if we're using SQR/ASQR
parser.add_argument('--query_cost', type=float, default=1.)
parser.add_argument('--n_max', type=int, default=0)
parser.add_argument('--normalize_rewards', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--num_R_samples', type=int, default=1)
parser.add_argument('--environment', type=str, default='chain5')
parser.add_argument('--agent', type=str, default='PSRLLimitedQuery')
parser.add_argument('--algorithm', type=str, default='ASQR')
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace
settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
save_str = os.path.join(save_dir, settings_str)
print "\n save_str=", save_str, '\n'

# ENVIRONMENT
if environment == 'grid1':
    grid_width = 8
    epLen = 2 * grid_width - 1
    prob_zero_reward=.9
    nAction = 5
    states = range(grid_width**2)
    reward_probabilities = [0,] * len(states[:-1]) + [1,]
    env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
elif environment == 'sparse_grid':
    grid_width = 8
    epLen = 2 * grid_width - 1
    prob_zero_reward=.9
    nAction = 5
    states = range(grid_width**2)
    reward_probabilities = np.load(os.environ['HOME'] + '/TabulaRL/fixed_mdp0.npy')
    env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
elif environment == 'chain5':
    chain_len = 5
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-chain_len)
elif environment == 'chain10':
    chain_len = 10
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-chain_len)
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)


# AGENT
if agent == 'PSRLLimitedQuery':
    alg = finite_tabular_agents.PSRLLimitedQuery
elif agent == 'PSRL':
    alg = finite_tabular_agents.PSRL
initial_agent = alg(env.nState, env.nAction, env.epLen, P_true=None, R_true=None)

#
num_episodes_remaining = num_episodes
# estimate the best n in each sampled environment (and for each c)
if algorithm == 'SQR':
    best_ns = []
    for sample_n in range(num_R_samples):
        best_ns.append(run_SQR(initial_agent, n_max, env, num_episodes_remaining, query_cost=query_cost, num_R_samples=num_R_samples, normalize_rewards=normalize_rewards))
    np.save(save_str + '_best_ns', best_ns)
# OR: compute cost and returns for each value of n
elif algorithm == 'ASQR':
    _, returns, max_returns, min_returns = run_ASQR(initial_agent, n_max, num_episodes_remaining, query_cost=query_cost, num_R_samples=num_R_samples, normalize_rewards=normalize_rewards)
    np.save(save_str + '_returns', returns)
    np.save(save_str + '_returns_max_and_min', [max_returns] + [min_returns])
elif algorithm == 'fixed_n':
    for n in range(n_max + 1):
        query_function = query_functions.QueryFirstNVisits(query_cost, n)
        agent = alg(env.nState, env.nAction, env.epLen,
                                  P_true=None, R_true=None,
                                  query_function=query_function)
        query_function.setAgent(agent)
        # Run the experiment
        result = run_finite_tabular_experiment(agent, env, f_ext, num_episodes, seed,
                            recFreq=1000, fileFreq=1000, targetPath=save_str + '_fixed_n=' + str(n), query_function=query_function,
                            printing=0)

