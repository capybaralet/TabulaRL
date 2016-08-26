import numpy as np
import argparse
from collections import defaultdict

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
#from experiment import run_finite_tabular_experiment
#from TabulaRL.dk_run_finite_tabular_experiment import run_finite_tabular_experiment
from TabulaRL.environment import make_stochasticChain
#np.random.seed(1)

from ASQR_and_SQR import *

import time
t1 = time.time()

# TODO (later):
#   ASQR in the loop
#   log visit/query counts, desired query sets

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

# THINGS TO DEAL WITH JOHN's GRIDWORLD mod:

# modify an agent's prior so that it knows that only the stay actions provide reward
def modifyPrior(prior): 
   def nonStayKnown(sa, p ): 
       #non 'stay' actions have infinite precision
       _, action = sa
       mu, tau = p
       if action != 0: 
           return (mu, 1e11)
       else: 
           return (mu, tau)
   return { k : nonStayKnown(k, v) for k,v in prior.iteritems() }

# modify query function to not query specific state action pairs (i.e. any actions other than stay action)
# e.g. query_function=QueryFixedFunction(query_cost, lambda s,a: (a==0) * n)
class QueryFixedFunction(query_functions.QueryFunction):
    def __init__(self, queryCost, func):
        self.__dict__.update(locals())
        self.visit_count = defaultdict(lambda :0)

    def __call__(self, state, action, episode, timestep):
        query = self.will_query(state, action)
        self.visit_count[state, action] += 1
        return query, query*self.queryCost

    #We can rewrite all query functions to use this subroutine when called
    def will_query(self, state, action):
        return self.visit_count[state, action] < self.func(state, action)

# end USEFUL FUNCTIONS
#-----------------------------------------------------------------------------------


# SETUP
import argparse
parser = argparse.ArgumentParser()
# we only need different costs if we're using ASQR
#parser.add_argument('--query_cost', type=float, default=1.)
query_cost = 1.
parser.add_argument('--log_n_max', type=int, default=10)
#parser.add_argument('--normalize_rewards', type=int, default=0)
normalize_rewards=0
parser.add_argument('--log_num_episodes', type=int, default=15)
parser.add_argument('--num_R_samples', type=int, default=100)
parser.add_argument('--environment', type=str, default='chain5')
parser.add_argument('--agent', type=str, default='PSRLLimitedQuery')
parser.add_argument('--algorithm', type=str, default='fixed_n')
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

assert query_cost > 0

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



# RUN
initial_env = env
num_episodes_remaining = 2**log_num_episodes
n_max = 2**log_n_max
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))

# save results here:
num_queries = np.empty((num_R_samples, log_num_episodes+1, log_n_max+1))
returns = np.empty((num_R_samples, log_num_episodes+1, log_n_max+1))
returns_max_min = np.empty((num_R_samples, 2))


if 1:
    for kk in range(num_R_samples):
        print "beginning experiment #", kk
        env = copy.deepcopy(initial_env)

        if algorithm in ['SQR', 'ASQR']: # use a sampled environment
            sampled_R, sampled_P = initial_agent.sample_mdp()
            env.R = {kk:(sampled_R[kk], variance_of_simulated_queries) for kk in sampled_R}
            env.P = sampled_P
            # TODO: use P_true or sampled_P here? (for now they are the same...)
            returns_max_min[kk,0] = initial_agent.compute_qVals(sampled_R, sampled_P)[1][0][0]
            returns_max_min[kk,1] = - initial_agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, sampled_P)[1][0][0]

        sampled_rewards = {(s,a) : sample_gaussian(env.R[s,a][0], env.R[s,a][1], n_max) for (s,a) in env.R.keys()}
        # is this still needed??
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        for ind, n in enumerate(ns):
            agent = copy.deepcopy(initial_agent)
            if environment.startswith('grid'):
                query_function = QueryFixedFunction(query_cost, lambda s,a: (a==0) * n)
            else:
                query_function = query_functions.QueryFirstNVisits(query_cost, n)
            query_function.setAgent(agent)

            if algorithm=="ASQR": # update posterior, compute expected returns
                updated_R = {}
                for [s,a] in first_n_sampled_rewards[n]: # TODO: clean-up (make a separate function?)
                    mu0, tau0 = agent.R_prior[s,a]
                    num_samples = len(first_n_sampled_rewards[n][s,a])
                    tau1 = tau0 + agent.tau * num_samples
                    mu1 = (mu0 * tau0 + sum(first_n_sampled_rewards[n][s,a]) * agent.tau) / tau1
                    updated_R[s,a] = mu1
                updated_P = sampled_P # TODO: use ASQR to learn about P somehow??
                expected_returns = agent.compute_qVals_true(updated_R, updated_P, sampled_R, sampled_P)[0]
                returns[kk, :, ind] = expected_returns * 2**np.arange(log_num_episodes+1)
                num_queries[kk, :, ind] = n * sum([agent.query_function.will_query(s,a) for [s,a] in first_n_sampled_rewards[n]])
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
                        if query and first_n_sampled_rewards[n] is not None:
                            reward = first_n_sampled_rewards[n][oldState, action][agent.query_function.visit_count[oldState, action] - 1]
                        cumReward += reward 
                        agent.update_obs(oldState, action, reward, newState, pContinue, h, query)

                    if is_power2(ep): # checkpoint
                        returns[kk, int(np.log2(ep)), ind] = cumReward
                        num_queries[kk, int(np.log2(ep)), ind] = cumQueryCost / query_cost

                # ---------------------------------------------------------------------
        np.save(save_str + 'num_queries', num_queries)
        np.save(save_str + 'returns', returns)
        np.save(save_str + 'returns_max_min', returns_max_min)


