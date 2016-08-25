import numpy as np
import argparse
import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
#from experiment import run_finite_tabular_experiment
from TabulaRL.dk_run_finite_tabular_experiment import run_finite_tabular_experiment
from TabulaRL.environment import make_stochasticChain
#np.random.seed(1)

from ASQR_and_SQR import *

import time
t1 = time.time()

# TODO (now): 
#   save results at different episodes
#   use powers of 2 for n, num_episodes
#
# TODO (later):
#   ASQR in the loop
#   log visit/query counts, desired query sets


#-----------------------------------------------------------------------------------
# UTILITY FUNCTIONS

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

# end UTILITY FUNCTIONS
#-----------------------------------------------------------------------------------


# SETUP
import argparse
parser = argparse.ArgumentParser()
# we only need different costs if we're using SQR/ASQR
parser.add_argument('--query_cost', type=float, default=1.)
parser.add_argument('--n_max', type=int, default=0)
parser.add_argument('--normalize_rewards', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=100000)
parser.add_argument('--num_R_samples', type=int, default=1)
parser.add_argument('--environment', type=str, default='chain5')
parser.add_argument('--agent', type=str, default='PSRLLimitedQuery')
parser.add_argument('--algorithm', type=str, default='ASQR')
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

assert query_cost > 0

settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])

# TODO: save results in a single file / database
import os
filename = os.path.basename(__file__)
save_dir = os.path.join(os.environ['HOME'], 'TabulaRL/src/results/results__' + filename)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

import datetime
timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
save_dir += '/' + timestamp + '___' + settings_str
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_str = save_dir + '/'
print "\n save_str=", save_str, '\n'


# ENVIRONMENT
if environment == 'grid1':
    grid_width = 8
    epLen = 2 * grid_width - 1
    prob_zero_reward=.9
    nAction = 5
    states = range(grid_width**2)
    reward_probabilities = [0,] * len(states[:-1]) + [1,]
    #reward_probabilities = gridworld.reward_for_action(reward_probabilities, action=0)
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



# RUN
num_episodes_remaining = num_episodes

"""
Steps of running an algorithm:
    set-up things to save results
    loop:
        sample R~, r~
        run an experiment (just copy code from run_finite_tabul...)
"""

num_queries = []
returns = []
returns_max = []
returns_min = []

if algorithm == 'ASQR': # runs ~num_episodes/2 times faster than the others!!
    _, returns, max_returns, min_returns = run_ASQR(initial_agent, n_max, num_episodes_remaining, query_cost=query_cost, num_R_samples=num_R_samples, normalize_rewards=normalize_rewards)
    np.save(save_str + 'returns', returns)
    np.save(save_str + 'returns_max_and_min', [max_returns] + [min_returns])
elif algorithm == 'SQR':
    num_queries, returns, max_returns, min_returns = run_SQR(initial_agent, n_max, env, num_episodes_remaining, query_cost=query_cost, num_R_samples=num_R_samples, normalize_rewards=normalize_rewards)
    np.save(save_str + 'num_queries', num_queries)
    np.save(save_str + 'returns', returns)
    np.save(save_str + 'returns_max_and_min', [max_returns] + [min_returns])
elif algorithm == 'fixed_n':
    returnss = []
    #visit_countss = []
    num_queriess = []
    #while time.time() - t1 < 60*60: # run for 1 hour
    for kk in range(num_R_samples):
        returns = []
        num_queries = []
        sampled_rewards = {(s,a) : sample_gaussian(env.R[s,a][0], env.R[s,a][1], n_max) for (s,a) in env.R.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        for n in range(n_max + 1):
            agent = copy.deepcopy(initial_agent)
            # TODO: similar thing for SQR/ASQR...
            if 0:#environment.startswith('grid'):
                query_function = QueryFixedFunction(query_cost, lambda s,a: (a==0) * n)
            else:
                query_function = query_functions.QueryFirstNVisits(query_cost, n)
            query_function.setAgent(agent)
            # Run an experiment
            result = run_finite_tabular_experiment(agent, env, f_ext, num_episodes,
                                recFreq=1000, fileFreq=1000, targetPath=save_str + '_fixed_n=' + str(n),
                                sampled_rewards=first_n_sampled_rewards[n],
                                printing=0, saving=0)
            returns.append(result[0])
            num_queries.append(result[1] / query_cost)
            #visit_counts.append(agent.query_function.visit_count)
        returnss.append(returns)
        num_queriess.append(num_queries)
    np.save(save_str + 'num_queries', num_queries)
    np.save(save_str + 'returns', returns)


