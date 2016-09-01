import numpy as np
import argparse
from collections import defaultdict
import copy

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.environment import make_stochasticChain, make_deterministicChain
#np.random.seed(1)

from generic_functions import add_dicts, dict_argmax, is_power2, sample_gaussian, update_gaussian_posterior_mean

import time
t1 = time.time()

# TODO: more logging, e.g. visit/query counts, desired query sets
# TODO: don't use env as a variable name!

"""
For now, I'm running OwainPSRL(_tilde)

"""

#-----------------------------------------------------------------------------------
# SETUP
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_n_max', type=int, default=10)
parser.add_argument('--log_num_episodes', type=int, default=10)
num_env_samples=1 # TODO: argparse?
parser.add_argument('--num_exps', type=int, default=1)
parser.add_argument('--update_freq', type=int, default=1)
parser.add_argument('--query_cost', type=float, default=1.)
parser.add_argument('--enviro', type=str, default='det_chain6')
parser.add_argument('--query_fn_selector', type=str, default='ASQR')#, options=['ASQR', 'SQR', 'OwainPSRL', 'Jan'])
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
    chain_len = int(enviro.split('chain')[1])
    epLen = chain_len
    env = make_deterministicChain(chain_len, chain_len)
elif enviro.startswith('stoch_chain'):
    chain_len = int(enviro.split('chain')[1])
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)


# AGENT
alg = finite_tabular_agents.PSRLLimitedQuery
initial_agent = alg(env.nState, env.nAction, env.epLen, P_true=None, R_true=None)
query_function = query_functions.QueryFirstNVisits(query_cost, n=0)
query_function.setAgent(initial_agent)


# QUERY FUNCTION SELECTOR
if query_fn_selector == 'ASQR':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count):
        perfs = np.empty((len(sampled_envs), len(ns)))
        for ii, sampled_env in enumerate(sampled_envs):
            for jj, n in enumerate(ns):
                sampled_rewards = {(s,a) : sample_gaussian(sampled_env.R[s,a][0], sampled_env.R[s,a][1], n_max) for (s,a) in sampled_env.R.keys()}
                updated_R = {sa: update_gaussian_posterior_mean(agent.R_prior[sa], sampled_rewards[sa][visit_count[sa]:n], agent.tau) for sa in sampled_rewards}
                updated_P = sampled_env.P
                expected_returns = agent.compute_qVals_true(updated_R, updated_P, {sa: sampled_env.R[sa][0] for sa in sampled_env.R}, sampled_env.P)[0]
                perfs[ii,jj] = neps * expected_returns - query_cost * sum([n - visit_count[sa] for sa in sampled_rewards]) #n * len([sa for sa in sampled_rewards])
        return query_functions.QueryFirstNVisits(query_cost, ns[np.argmax(perfs.mean(0))])

# VoI = E * (R(pi+, M+) - R(pi, M+))
#
# OLD NOTES:
# Here, we compute the rewards in the agent's MLE env    (potentially with the sampled r(sa) replacing the MLE r(sa)) 
# We also discussed computing rewards in the sampled_env (potentially with the MLE r(sa) replacing the sampled r(sa))
# More generally, we could look at various ways of interpolating btw the two...
# Is there some analogy with using the UCBs for planning?
# TODO:
elif query_fn_selector == 'OwainPSRL':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count):
        assert len(sampled_envs) == 1
        sampled_env = sampled_envs[0]
        VoIs = defaultdict(lambda : [])
        R, P = agent.map_mdp()
        sampled_R, sampled_P = sampled_env.R, sampled_env.P
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
        num_queries = {sa: visit_count[sa] + sum([(VoIs[sa] > query_cost * nn) for nn in range(max(ns))]) for sa in VoIs}
        return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s, a])

# VoI = E * (R(pi+, M~) - R(pi, M~)) <-- not necessarily positive :/
elif query_fn_selector == 'OwainPSRL_tilde':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count):
        assert len(sampled_envs) == 1
        sampled_env = sampled_envs[0]
        VoIs = defaultdict(lambda : [])
        R, P = agent.map_mdp()
        sampled_R, sampled_P = sampled_env.R, sampled_env.P
        sampled_R = {sa: sampled_R[sa][0] for sa in sampled_R}
        for sa in agent.R_prior:
            # create M+
            updated_R = copy.deepcopy(R)
            updated_R[sa] = sampled_R[sa]
            updated_P = sampled_P
            # compute Rs
            expected_return_ignorant = agent.compute_qVals_true(R, P, sampled_R, sampled_P)[0]
            expected_return_informed = agent.compute_qVals_true(updated_R, updated_P, sampled_R, sampled_P)[0]
            return_diff = expected_return_informed - expected_return_ignorant
            VoIs[sa] = neps * return_diff
        # compare avg_VoI to query cost, and plan to query up to n times (more)
        num_queries = {sa: visit_count[sa] + sum([(VoIs[sa] > query_cost * nn) for nn in range(max(ns))]) for sa in VoIs}
        return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s, a])


# FIXME: we also want to change the PLANNING! (although that's just doing PSRL, no?)
#   if sampled_envs == 1, then we can just sample once (right?)
#   What if we sample just all sa simultaneously in the sampled environment, and add their VoI??
#   NOT SURE WHAT I'M DOING HERE....
elif query_fn_selector == 'OwainPSRL_multiple_envs':
    def query_function_selector(agent, sampled_envs, neps, query_cost, ns, visit_count):
        # how much value is there to knowing the r_sa?
        VoIs = defaultdict(lambda : [])
        # TODO: check it: wasn't I supposed to use ii??
        for ii, sampled_env in enumerate(sampled_envs):
            R, P = agent.map_mdp()
            sampled_R, sampled_P = sampled_env.R, sampled_env.P
            expected_return_ignorant = agent.compute_qVals_true(R, P, {sa: sampled_R[sa][0] for sa in sampled_R}, sampled_P)[0]
            for sa in agent.R_prior:
                # for each sampled_env, M, we THEN sample a reward value, r_sa for each sa and look at the VoI of knowing that r(sa) = r_sa in M
                updated_R = {sa: sampled_R[sa][0] for sa in sampled_R}
                updated_R[sa] = sample_gaussian(sampled_env.R[sa][0], sampled_env.R[sa][1], 1)#[0]
                updated_P = sampled_P
                expected_return_informed = agent.compute_qVals_true(updated_R, updated_P, {sa: sampled_env.R[sa][0] for sa in sampled_env.R}, sampled_env.P)[0]
                return_diff = expected_return_informed - expected_return_ignorant
                VoIs[sa].append(neps * return_diff)
        avg_VoIs = {sa: np.mean(VoIs[sa]) for sa in VoIs}
        # compare avg_VoI to query cost, and plan to query up to n times (more)
        num_queries = {sa: visit_count[sa] + sum([(avg_VoIs[sa] > query_cost * nn) for nn in range(max(ns))]) for sa in avg_VoIs}
        return query_functions.QueryFixedFunction(query_cost, lambda s, a: num_queries[s, a])
else:
    print "not implemented!"
    assert False 




#-----------------------------------------------------------------------------------
# RUN
initial_env = env
num_episodes= 2**log_num_episodes
num_updates = num_episodes / update_freq + 1
n_max = 2**log_n_max
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))



"""
ALSO: 
    think about better names for (A)SQR??
    think about other algos...
        how to chose query order for ASQR?

What do I want to log??
What do I want to run??

So... in previous experiments, (A)SQR reliably achieved near-optimal performance when we took enough samples (which is kind of surprising!)
So that means we need to find something harder to demonstrate that updating in the loop helps that much.
HOW TO SOLVE THIS:
    1. Search over more specific ns (so its harder to find the exact right one)
    2. Use different envs
        reachability
    3. Look at wall-clock time?
    4.

We also expect that (A)SQR doesn't work in some environments (because of, e.g. reachability), and we should demonstrate that.
    

"""

# record results here:
# TODO: change for intheloop  (extra dim: num_updates?)
num_queries = np.empty((num_exps, log_num_episodes+1))
returns = np.empty((num_exps, log_num_episodes+1))
returns_max_min = np.empty((num_exps, 2))

for kk in range(num_exps): # run an entire exp
    print "beginning exp #", kk
    # TODO: rm?
    env = copy.deepcopy(initial_env)
    # TODO: the total number of queries of a given state may be higher than n_max, now
    sampled_rewards = {(s,a) : sample_gaussian(env.R[s,a][0], env.R[s,a][1], n_max*num_episodes) for (s,a) in env.R.keys()}
    agent = copy.deepcopy(initial_agent)

    visit_count = defaultdict(lambda : 0)
    cumReward = 0
    cumQueryCost = 0 
    for ep in xrange(1, num_episodes + 2):
        print ep

        # UPDATE query function?
        # (For now, we just update periodically.)
        if (ep-1) % update_freq == 0:
            # TODO: log query_functions desired_query_sets
            sampled_envs = []
            for n_env in range(num_env_samples): # sample a new environment
                sampled_env = copy.deepcopy(initial_env)
                sampled_R, sampled_P = agent.sample_mdp()
                sampled_env.R = {kk:(sampled_R[kk], 1) for kk in sampled_R}
                sampled_env.P = sampled_P
                sampled_envs.append(sampled_env)
                returns_max_min[kk,0] = agent.compute_qVals(sampled_R, sampled_P)[1][0][0]
                returns_max_min[kk,1] = - agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, sampled_P)[1][0][0]
            query_function = query_function_selector(agent, sampled_envs, num_episodes - ep + 1, query_cost, ns, visit_count)
            query_function.visit_count = visit_count
            query_function.setAgent(agent)

        # RUN some episodes with the current query function:
        env.reset() # return to tstep=state=0
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
                reward = sampled_rewards[oldState, action][visit_count[oldState, action] - 1]
            cumReward += reward 
            agent.update_obs(oldState, action, reward, newState, pContinue, h, query)

        # CHECKPOINT (TODO)
        if is_power2(ep):
            returns[kk, int(np.log2(ep))] = cumReward
            num_queries[kk, int(np.log2(ep))] = cumQueryCost / query_cost

            # ---------------------------------------------------------------------
    if save:
        np.save(save_str + 'num_queries', num_queries)
        np.save(save_str + 'returns', returns)
        np.save(save_str + 'returns_max_min', returns_max_min)

if save:
    os.system('touch' + save_str + 'FINISHED')

