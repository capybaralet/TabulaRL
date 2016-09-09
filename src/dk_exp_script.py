import numpy as np
import argparse
from collections import defaultdict

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.environment import make_stochasticChain, make_deterministicChain
#np.random.seed(1)

from generic_functions import dict_argmax, is_power2, sample_gaussian, update_gaussian_posterior_mean

import time
t1 = time.time()

# TODO: more logging, e.g. visit/query counts, desired query sets
# TODO: don't use env as a variable name!

#-----------------------------------------------------------------------------------
# SETUP
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_n_max', type=int, default=3)
parser.add_argument('--log_num_episodes', type=int, default=4)
parser.add_argument('--num_exps', type=int, default=1)
parser.add_argument('--enviro', type=str, default='det_chain6')
parser.add_argument('--query_fn', type=str, default='fixed_n')
# not included in save_str:
parser.add_argument('--save', type=str, default=0)
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

if args_dict.pop('save'):
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
if enviro.startswith('grid'):
    grid_width = int(enviro.split('grid')[1])
    epLen = 2 * grid_width - 1
    nAction = 5
    states = range(grid_width**2)
    reward_means = np.diag(np.linspace(0,1,grid_width)).flatten()
    env = gridworld.make_gridworld(grid_width, epLen, gridworld.make_sa_rewards(reward_means))
elif enviro.startswith('det_chain'):
    chain_width = int(enviro.split('chain')[1])
    epLen = chain_len
    #env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
    env = make_deterministicChain(chain_len, max_reward=1)
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)


# AGENT
alg = finite_tabular_agents.PSRLLimitedQuery
initial_agent = alg(env.nState, env.nAction, env.epLen, P_true=None, R_true=None)



# RUN

"""
To do things in the loop, we'll need to:
    specify the query_cost, horizon
    run update_query_fn algorithm (may involve making another agent/env copy...)
        repeat until ___:
            sample env
              sample rewards (potentially)
            evaluate query function in sampled env 
        return a query function
        update agent

We'll start with ASQR, since it's already implemented.
Then we'll do Owain's thing (NAME IT!) since it's already well specified
Then Jan's thing (NAME IT!)

ALSO: think about better names for (A)SQR??

"""
initial_env = env
num_episodes_remaining = 2**log_num_episodes
n_max = 2**log_n_max
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))
query_cost = 1.

# record results here:
num_queries = np.empty((num_experiments, log_num_episodes+1, log_n_max+1))
returns = np.empty((num_experiments, log_num_episodes+1, log_n_max+1))
returns_max_min = np.empty((num_experiments, 2))


for kk in range(num_experiments):
    print "beginning experiment #", kk
    env = copy.deepcopy(initial_env)

    if query_fn in ['SQR', 'ASQR']: # use a sampled enviro
        sampled_R, sampled_P = initial_agent.sample_mdp()
        env.R = {kk:(sampled_R[kk], agent.tau) for kk in sampled_R}
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
        if enviro.startswith('grid'):
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
    if save:
        np.save(save_str + 'num_queries', num_queries)
        np.save(save_str + 'returns', returns)
        np.save(save_str + 'returns_max_min', returns_max_min)

if save:
    os.system('touch' + save_str + 'FINISHED')
