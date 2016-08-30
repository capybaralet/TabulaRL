import numpy as np
import argparse
from collections import defaultdict

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.environment import make_stochasticChain
#np.random.seed(1)

from ASQR_and_SQR import *

import time
t1 = time.time()

# TODO (later):
#   ASQR in the loop
#   log visit/query counts, desired query sets

# TODO: don't use env as a variable name!
# TODO: chain max_reward HACK: increasing max_reward effects the validity of the prior!!!!
#   Use deterministic chain!

#-----------------------------------------------------------------------------------
# USEFUL FUNCTIONS

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

#-----------------------------------------------------------------------------------
# SETUP
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_n_max', type=int, default=10)
parser.add_argument('--log_num_episodes', type=int, default=15)
parser.add_argument('--num_R_samples', type=int, default=100)
parser.add_argument('--environment', type=str, default='chain5')
parser.add_argument('--agent', type=str, default='PSRLLimitedQuery')
parser.add_argument('--query_fn', type=str, default='fixed_n')
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])

# TODO: save results in a single file / database
import os
filename = os.path.basename(__file__)
save_dir = os.path.join(os.environ['HOME'], 'TabulaRL/src/results/results__' + filename)

import datetime
timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
save_dir += '/' + timestamp + '___' + settings_str
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_str = save_dir + '/'
print "\n save_str=", save_str, '\n'


# ENVIRONMENT
if environment == 'grid8':
    grid_width = 8
    epLen = 2 * grid_width - 1
    prob_zero_reward=.9
    nAction = 5
    states = range(grid_width**2)
    reward_probabilities = np.load(os.environ['HOME'] + '/TabulaRL/fixed_mdp0.npy')
    env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
elif environment == 'grid4':
    grid_width = 4
    epLen = 2 * grid_width - 1
    prob_zero_reward=.8
    nAction = 5
    states = range(grid_width**2)
    reward_probabilities = np.load(os.environ['HOME'] + '/TabulaRL/fixed_grid4.npy')
    env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
elif environment == 'chain5':
    chain_len = 5
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
elif environment == 'chain10':
    chain_len = 10
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)


# AGENT
if agent == 'PSRLLimitedQuery':
    alg = finite_tabular_agents.PSRLLimitedQuery
elif agent == 'PSRL':
    alg = finite_tabular_agents.PSRL
initial_agent = alg(env.nState, env.nAction, env.epLen, P_true=None, R_true=None)



# RUN
initial_env = env
num_episodes_remaining = 2**log_num_episodes
n_max = 2**log_n_max
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))
query_cost = 1.

# save results here:
num_queries = np.empty((num_R_samples, log_num_episodes+1, log_n_max+1))
returns = np.empty((num_R_samples, log_num_episodes+1, log_n_max+1))
returns_max_min = np.empty((num_R_samples, 2))


for kk in range(num_R_samples):
    print "beginning experiment #", kk
    env = copy.deepcopy(initial_env)

    if query_fn in ['SQR', 'ASQR']: # use a sampled environment
        sampled_R, sampled_P = initial_agent.sample_mdp()
        env.R = {kk:(sampled_R[kk], 1) for kk in sampled_R}
        env.P = sampled_P
        returns_max_min[kk,0] = initial_agent.compute_qVals(sampled_R, sampled_P)[1][0][0]
        returns_max_min[kk,1] = - initial_agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, sampled_P)[1][0][0]
    else:
        mean_rewards = {kk:env.R[kk][0] for kk in env.R}
        returns_max_min[kk,0] = initial_agent.compute_qVals(mean_rewards, env.P)[1][0][0]
        returns_max_min[kk,1] = - initial_agent.compute_qVals({kk: -mean_rewards[kk] for kk in mean_rewards}, env.P)[1][0][0]

    sampled_rewards = {(s,a) : sample_gaussian(env.R[s,a][0], env.R[s,a][1], n_max) for (s,a) in env.R.keys()}
    for ind, n in enumerate(ns):
        agent = copy.deepcopy(initial_agent)
        if environment.startswith('grid'):
            query_function = query_functions.QueryFixedFunction(query_cost, lambda s,a: (a==0) * n)
            agent.R_prior = finite_tabular_agents.modifyPrior(agent.R_prior)
        else:
            query_function = query_functions.QueryFirstNVisits(query_cost, n)
        query_function.setAgent(agent)

        if query_fn=="ASQR": # update posterior, compute expected returns
            # just the mean
            updated_R = {sa: update_gaussian_posterior_mean(agent.R_prior[sa], sampled_rewards[sa][:n], agent.tau) for sa in sampled_rewards}
            updated_P = sampled_P # TODO: use ASQR to learn about P!
            expected_returns = agent.compute_qVals_true(updated_R, updated_P, sampled_R, sampled_P)[0]
            returns[kk, :, ind] = expected_returns * 2**np.arange(log_num_episodes+1)
            num_queries[kk, :, ind] = n * sum([agent.query_function.will_query(s,a) for [s,a] in sampled_rewards])
        else: # Run an experiment 
            nEps = num_episodes_remaining
            # --------------- modified from dk_run_finite_tabular_experiment ------------------
            qVals, qMax = env.compute_qVals()
            cumReward = 0
            cumQueryCost = 0 
            for ep in xrange(1, nEps + 2):
                env.reset()
                epMaxVal = qMax[env.timestep][env.state]
                agent.update_policy(ep)

                pContinue = 1
                while pContinue > 0:
                    # Step through the episode
                    h, oldState = f_ext.get_feat(env)

                    action = agent.pick_action(oldState, h)
                    query, queryCost = agent.query_function(oldState, action, ep, h)
                    cumQueryCost += queryCost

                    reward, newState, pContinue = env.advance(action)
                    if query:
                        reward = sampled_rewards[oldState, action][agent.query_function.visit_count[oldState, action] - 1]
                    cumReward += reward 
                    agent.update_obs(oldState, action, reward, newState, pContinue, h, query)

                if is_power2(ep): # checkpoint
                    returns[kk, int(np.log2(ep)), ind] = cumReward
                    num_queries[kk, int(np.log2(ep)), ind] = cumQueryCost / query_cost

            # ---------------------------------------------------------------------
    if 1:
        np.save(save_str + 'num_queries', num_queries)
        np.save(save_str + 'returns', returns)
        np.save(save_str + 'returns_max_min', returns_max_min)


