import numpy as np
import argparse
import itertools
import copy

from collections import defaultdict

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.environment import make_stochasticChain, make_deterministicChain
#np.random.seed(1)

from generic_functions import add_dicts, dict_argmax, is_power2, sample_gaussian, update_gaussian_posterior_mean, pysave

import time
t1 = time.time()

# TODO: clean-up logging

#-----------------------------------------------------------------------------------
# SETUP
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--log_n_max', type=int, default=10)
log_n_max=10
parser.add_argument('--log_num_episodes', type=int, default=6)
num_env_samples=1
parser.add_argument('--num_exps', type=int, default=1)
parser.add_argument('--update_freq', type=int, default=1)
#update_freq = 1
parser.add_argument('--query_cost', type=float, default=1.)
parser.add_argument('--reward_noise', type=float, default=1.)
parser.add_argument('--enviro', type=str, default='det_chain6')
parser.add_argument('--query_fn_selector', type=str, default='VOI_PSRL_greedy')
# not included in save_path:
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--printing', type=int, default=0)
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

num_episodes= 2**log_num_episodes
normalize_rewards = False

save = args_dict.pop('save')
printing = args_dict.pop('printing')

import datetime
timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
print timestamp

if save:
    if save_path is None:
        settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])

        # TODO: save results in a single file / database
        import os
        filename = os.path.basename(__file__)
        #save_dir = os.path.join(os.environ['HOME'], 'TabulaRL/src/results/results__' + filename)
        save_path = os.path.join(os.environ['SAVE_PATH'], 'TabulaRL/' + filename)
        save_path += '/' + timestamp + '___' + settings_str + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print "\n save_path=", save_path, '\n'

#assert False


# ENVIRONMENT
if enviro.startswith('grid'):
    grid_width = int(enviro.split('grid')[1])
    epLen = 2 * grid_width - 1
    reward_means = np.diag(np.linspace(0, 1, grid_width)).flatten()
    envv = gridworld.make_gridworld(grid_width, epLen, gridworld.make_sa_rewards(reward_means, actions = range(5)),
                                     multi_chain=False, gotta_move=True, reward_noise=reward_noise)
# TODO: leaving the extra states around messes with ASQR!
# FIXME: the agent should have to dig!
elif enviro.startswith('multi_chain'):
    grid_width = int(enviro.split('multi_chain')[1])
    epLen = 2 * grid_width - 1 # agent needs one extra move (only gets reward in a state if it acts in it!)
    reward_means = np.diag(np.linspace(0, 1, grid_width)).flatten()
    envv = gridworld.make_gridworld(grid_width, epLen, gridworld.make_sa_rewards(reward_means, actions = range(5)), 
                                     multi_chain=True, gotta_move=True)
elif enviro.startswith('det_chain'):
    chain_len = int(enviro.split('det_chain')[1])
    epLen = chain_len
    envv = make_deterministicChain(chain_len, epLen)
elif enviro.startswith('stoch_chain'):
    chain_len = int(enviro.split('stoch_chain')[1])
    epLen = chain_len
    envv = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
elif enviro.startswith('longY'):
    nState = int(enviro.split('longY')[1])
    epLen = nState - 1
    reward_means = np.zeros(nState)
    reward_means[-1] = 1
    envv = gridworld.make_longY(nState, epLen, 
                                 rewards=gridworld.make_sa_rewards(reward_means, actions=range(2)), reward_noise=reward_noise)
elif enviro.startswith('random'): # TODO
    nState = int(enviro.split('_')[1])
    num_next_states = int(enviro.split('_')[2])
    epLen = nState - 1
    reward_means = np.random.randn(nState)
    R = gridworld.make_sa_rewards(reward_means, actions=range(2))
    R = { k: (v, reward_noise) for k,v in R.iteritems() }
    P = gridworld.make_random_P(nState, num_next_states)
    envv = gridworld.make_mdp(nState, nAction=2, epLen=epLen, 
                              R=R, P=P)#, reward_noise=reward_noise)
else:
    assert False
f_ext = FeatureTrueState(envv.epLen, envv.nState, envv.nAction, envv.nState)


# AGENT
alg = finite_tabular_agents.PSRLLimitedQuery
if enviro.startswith('grid') or enviro.startswith('multi_chain') or enviro.startswith('longY'):
    initial_agent = alg(envv.nState, envv.nAction, envv.epLen, 
                         P_true=envv.P, R_true=None, reward_depends_on_action=False, tau=1/reward_noise**2, tau0=1/reward_noise**2)
else:
    initial_agent = alg(envv.nState, envv.nAction, envv.epLen, 
                         P_true=envv.P, R_true=None, tau=1/reward_noise**2, tau0=1/reward_noise**2)
query_function = query_functions.QueryFirstNVisits(query_cost, n=0)
query_function.setAgent(initial_agent)

def ASQR_query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
    perfs = np.empty((len(sampled_envs), len(ns)))
    for ii, sampled_env in enumerate(sampled_envs):
        # TODO: should have a posterior over reward variance for this to make sense
        sampled_rewards = {(s,a) : sample_gaussian(sampled_env.R[s,a][0], sampled_env.R[s,a][1], n_max) for (s,a) in sampled_env.R.keys()}
        for jj, n in enumerate(ns):
            updated_R = {sa: update_gaussian_posterior_mean(agent.R_prior[sa], sampled_rewards[sa][visit_count[sa]:n], agent.tau) for sa in sampled_rewards}
            updated_P = sampled_env.P
            expected_returns = agent.compute_qVals_true(updated_R, updated_P, {sa: sampled_env.R[sa][0] for sa in sampled_env.R}, sampled_env.P)[0]
            if agent.reward_depends_on_action:
                perfs[ii,jj] = neps * expected_returns - query_cost * sum([n - query_count[sa] for sa in sampled_rewards]) #n * len([sa for sa in sampled_rewards])
            else:
                perfs[ii,jj] = neps * expected_returns - query_cost * sum([n - query_count[s] for s in range(agent.nState)]) #n * len([sa for sa in sampled_rewards])
            #import ipdb; ipdb.set_trace()
    return query_functions.QueryFirstNVisits(query_cost, ns[np.argmax(perfs.mean(0))])

# QUERY FUNCTION SELECTOR
# TODO: clean-up this stuff a lot (maybe move to separate script?)
# for fixed ones, we'll use 25 / sa
# query_function = query_function_selector(agent, sampled_envs, num_episodes - ep + 1, query_cost, ns, visit_count, query_count)
if query_fn_selector == 'fixed_firstNvisits':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        return query_functions.QueryFixedFunction(query_cost, lambda s, a: 2*log_num_episodes)
elif query_fn_selector.startswith('fixed_first'):
    strs = query_fn_selector.split('fixed_first')[1].split('visits')
    if len(strs) == 1:
        def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
            #return query_functions.QueryFirstN(query_cost, int(strs[0]))
            return query_functions.QueryFirstN(query_cost, int(strs[0])*envv.nState * (1 + agent.reward_depends_on_action * (envv.nAction - 1)))
    #elif strs[1] == 'per':
        #def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
            #return query_functions.QueryFirstN(query_cost, int(strs[0])*envv.nState * (1 + agent.reward_depends_on_action * (envv.nAction - 1)))
    else:
        def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
            return query_functions.QueryFirstNVisits(query_cost, int(strs[0]))
elif query_fn_selector == 'fixed_always':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        return query_functions.AlwaysQuery(query_cost )
elif query_fn_selector.startswith('fixed_decay'):
    max_query_prob = float(query_fn_selector.split('fixed_decay')[1])
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        return query_functions.DecayQueryProbability(query_cost, func=lambda e,t : max_query_prob * neps / num_episodes)

elif query_fn_selector in ['ASQR', 'fixed_ASQR']:
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        perfs = np.empty((len(sampled_envs), len(ns)))
        for ii, sampled_env in enumerate(sampled_envs):
            # TODO: should have a posterior over reward variance for this to make sense
            sampled_rewards = {(s,a) : sample_gaussian(sampled_env.R[s,a][0], sampled_env.R[s,a][1], n_max) for (s,a) in sampled_env.R.keys()}
            for jj, n in enumerate(ns):
                updated_R = {sa: update_gaussian_posterior_mean(agent.R_prior[sa], sampled_rewards[sa][visit_count[sa]:n], agent.tau) for sa in sampled_rewards}
                updated_P = sampled_env.P
                expected_returns = agent.compute_qVals_true(updated_R, updated_P, {sa: sampled_env.R[sa][0] for sa in sampled_env.R}, sampled_env.P)[0]
                if agent.reward_depends_on_action:
                    perfs[ii,jj] = neps * expected_returns - query_cost * sum([n - query_count[sa] for sa in sampled_rewards]) #n * len([sa for sa in sampled_rewards])
                else:
                    perfs[ii,jj] = neps * expected_returns - query_cost * sum([n - query_count[s] for s in range(agent.nState)]) #n * len([sa for sa in sampled_rewards])
                #import ipdb; ipdb.set_trace()
        return query_functions.QueryFirstNVisits(query_cost, ns[np.argmax(perfs.mean(0))])
    if query_fn_selector == 'fixed_ASQR':
        update_freq = np.inf

# FIXME: need to use the agent's prior knowledge that reward only depends on state (when it does)
elif query_fn_selector == 'VOI_PSRL_greedy':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        assert len(sampled_envs) == 1
        sampled_env = sampled_envs[0]
        VoIs = defaultdict(lambda : [])
        R, P = agent.map_mdp()
        sampled_R, sampled_P = sampled_env.R, sampled_env.P
        if agent.reward_depends_on_action:
            for sa in agent.R_prior:
                # create M+
                updated_R = copy.deepcopy(R)
                updated_R[sa] = sampled_R[sa][0]
                updated_P = sampled_P
                # compute Rs
                expected_return_ignorant = agent.compute_qVals_true(R, P, updated_R, updated_P)[0]
                expected_return_informed = agent.compute_qVals_true(updated_R, updated_P, updated_R, updated_P)[0]
                return_diff = expected_return_informed - expected_return_ignorant
                VoIs[sa] = neps * return_diff
            # compare avg_VoI to query cost, and plan to query up to n times (more)
            num_queries = {sa: query_count[sa] + sum([(VoIs[sa] >= query_cost * nn) for nn in range(1, max(ns))]) for sa in VoIs}
            return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s, a])
        else:
            for s in range(agent.nState):
                # create M+
                updated_R = copy.deepcopy(R)
                for action in range(agent.nAction):
                    updated_R[s, action] = sampled_R[s, 0][0]
                updated_P = sampled_P
                # compute Rs
                expected_return_ignorant = agent.compute_qVals_true(R, P, updated_R, updated_P)[0]
                expected_return_informed = agent.compute_qVals_true(updated_R, updated_P, updated_R, updated_P)[0]
                return_diff = expected_return_informed - expected_return_ignorant
                VoIs[s] = neps * return_diff
            # compare avg_VoI to query cost, and plan to query up to n times (more)
            num_queries = {s: query_count[s] + sum([(VoIs[s] >= query_cost * nn) for nn in range(1, max(ns))]) for s in VoIs}
            #import ipdb; ipdb.set_trace()
            return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s])

elif query_fn_selector == 'VOI_PSRL_omni':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count, query_count):
        assert len(sampled_envs) == 1
        sampled_env = sampled_envs[0]
        VoIs = defaultdict(lambda : [])
        R, P = agent.map_mdp()
        sampled_R, sampled_P = sampled_env.R, sampled_env.P
        updated_R = {sa: sampled_R[sa][0] for sa in sampled_R}
        updated_P = sampled_P
        expected_return_informed = agent.compute_qVals_true(updated_R, updated_P, updated_R, updated_P)[0]
        if agent.reward_depends_on_action:
            for sa in agent.R_prior:
                # create M~, M-
                R_minus = copy.deepcopy(updated_R)
                R_minus[sa] = R[sa]
                P_minus = sampled_P
                # compute Rs
                expected_return_ignorant = agent.compute_qVals_true(R_minus, P_minus, updated_R, updated_P)[0]
                return_diff = expected_return_informed - expected_return_ignorant
                VoIs[sa] = neps * return_diff
            # compare avg_VoI to query cost, and plan to query up to n times (more)
            num_queries = {sa: query_count[sa] + sum([(VoIs[sa] >= query_cost * nn) for nn in range(1, max(ns))]) for sa in VoIs}
            return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s, a])
        else:
            for s in range(agent.nState):
                # create M~, M-
                R_minus = copy.deepcopy(updated_R)
                for action in range(agent.nAction):
                    R_minus[s, action] = R[s, 0]
                P_minus = sampled_P
                # compute Rs
                expected_return_ignorant = agent.compute_qVals_true(R_minus, P_minus, updated_R, updated_P)[0]
                return_diff = expected_return_informed - expected_return_ignorant
                VoIs[s] = neps * return_diff
            # compare avg_VoI to query cost, and plan to query up to n times (more)
            num_queries = {s: query_count[s] + sum([(VoIs[s] >= query_cost * nn) for nn in range(1, max(ns))]) for s in VoIs}
            return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s])

else:
    print "not implemented!"
    assert False 


#-----------------------------------------------------------------------------------
# RUN
num_updates = num_episodes / update_freq + 1
n_max = 2**log_n_max
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))

# record results here:
num_queries = np.empty((num_exps, log_num_episodes+1))
returns = np.empty((num_exps, log_num_episodes+1))
exp_log = {}

for kk in range(num_exps): # run an entire exp
    print "beginning exp #", kk

    exp_log[kk] = {}

    sampled_rewards = {(s,a) : sample_gaussian(envv.R[s,a][0], envv.R[s,a][1], num_episodes*epLen) for (s,a) in envv.R.keys()}
    agent = copy.deepcopy(initial_agent)

    #visit_count = {sa: 0 for sa in itertools.product(range(envv.nState), range(envv.nAction))}
    #query_count = {sa: 0 for sa in itertools.product(range(envv.nState), range(envv.nAction))}
    query_function = query_functions.QueryFirstNVisits(query_cost, n=0)
    query_function.setAgent(initial_agent)
    cumReward = 0
    cumQueryCost = 0 

    for ep in xrange(1, num_episodes + 1):
        if printing:
            print ep

        # UPDATE query function?
        # (For now, we just update periodically.)
        if (ep-1) % update_freq == 0:
            visit_count = query_function.visit_count
            query_count = query_function.query_count
            sampled_envs = []
            for n_env in range(num_env_samples): # sample a new environment
                sampled_env = copy.deepcopy(envv)
                sampled_R, sampled_P = agent.sample_mdp_unclamped()
                if normalize_rewards: # TODO: account for uncertainty in P
                    returns_max = agent.compute_qVals(sampled_R, sampled_P)[1][0][0]
                    returns_min = - agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, sampled_P)[1][0][0]
                    returns_diff = returns_max - returns_min
                    sampled_env.R = {kk: (returns_diff * (sampled_R[kk] - returns_min), agent.tau) for kk in sampled_R}
                else:
                    sampled_env.R = {kk:(sampled_R[kk], agent.tau) for kk in sampled_R}
                sampled_env.P = sampled_P
                sampled_envs.append(sampled_env)
            # choose a query function and update its visit_count and query_count
            query_function = query_function_selector(agent, sampled_envs, num_episodes - ep + 1, query_cost, ns, visit_count, query_count)
            query_function.setAgent(agent)
            query_function.visit_count = visit_count
            query_function.query_count = query_count

        # RUN some episodes with the current query function:
        envv.reset() # return to tstep=state=0
        agent.update_policy(ep)
        pContinue = 1
        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(envv)

            action = agent.pick_action(oldState, h)
            query, queryCost = agent.query_function(oldState, action, ep, h)
            cumQueryCost += queryCost

            reward, newState, pContinue = envv.advance(action)
            if query:
                reward = sampled_rewards[oldState, action][visit_count[oldState, action] - 1]
            cumReward += reward 
            agent.update_obs(oldState, action, reward, newState, pContinue, h, query)

        # CHECKPOINT
        if is_power2(ep):
            returns[kk, int(np.log2(ep))] = cumReward
            num_queries[kk, int(np.log2(ep))] = sum(query_count.values())#cumQueryCost / query_cost
            state_visits = np.array([sum([visit_count[key] for key in visit_count if key[0] == nn]) for nn in range(envv.nState)])
            # save for every experiment (not just the last)!
            exp_log[kk][ep] = {}
            exp_log[kk][ep]['returns'] = cumReward
            exp_log[kk][ep]['visit_count'] = copy.deepcopy(visit_count)
            exp_log[kk][ep]['state_visits'] = copy.deepcopy(state_visits)
            exp_log[kk][ep]['query_count'] = copy.deepcopy(query_count)
            if printing and (enviro.startswith('grid') or enviro.startswith('multi_chain')):
                import pylab
                pylab.imshow(state_visits.reshape((grid_width, grid_width)), cmap=pylab.cm.gray, interpolation='nearest')
                pylab.draw(); pylab.show()

            # ---------------------------------------------------------------------
    if save and kk % 10 == 0:
        np.save(save_path + 'num_queries', num_queries)
        np.save(save_path + 'returns', returns)
        pysave(save_path + 'exp_log', exp_log)

if save:
    # FIXME: why is this being created prematurely?
    os.system('touch ' + save_path + 'FINISHED')

