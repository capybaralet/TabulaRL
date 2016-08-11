
import numpy
np = numpy
import copy

from experiment import run_finite_tabular_experiment
from feature_extractor import FeatureTrueState
from query_functions import QueryFirstNVisits


# TODO: should be 1 (setting it lower for debugging)
variance_of_simulated_queries = 1.


# TODO: sample a different number of rewards for different (s,a) pairs
#def sample_rewards(agent, queries, sampled_R):
#   """ queries[s,a] = # of queries to perform on (s,a) """
#   return {(s,a) : np.random.normal(sampled_R[s,a], 1, queries[s,a]) for (s,a) in agent.P_prior.keys()}

# TODO: maintaining a posterior over the variance (or maybe just using Bernoullis?) should help a lot!

# TODO: optionally normalize rewards (or performance??) across different sampled_R
# TODO: docstrings

def estimate_performance_ASQR(agent, sampled_R, sampled_rewards, query_cost):
    """
    sampled_R: a dictionary of mean rewards for each (s,a) pair, sampled from the agent's posterior.
    sampled_rewards: a dictionary of (several) sampled rewards for each (s,a) pair, sampled from sampled_R
    """
    _, P_hat = agent.map_mdp()
    R_hat = agent.R_prior
    updated_R = {}
    for [s,a] in sampled_rewards:
        mu0, tau0 = R_hat[s,a]
        num_samples = len(sampled_rewards[s,a])
        tau1 = tau0 + agent.tau * num_samples
        mu1 = (mu0 * tau0 + sum(sampled_rewards[s,a]) * agent.tau) / tau1
        updated_R[s,a] = mu1
    return agent.compute_qVals_true(updated_R, P_hat, sampled_R, P_hat)[0] - query_cost * sum([len(qq) for qq in sampled_rewards.values()])
    #return compute_qVals_true(agent, updated_R, P_hat, sampled_R, P_hat)[0] - query_cost * sum([len(qq) for qq in sampled_rewards.values()])

def run_ASQR(agent, n_max, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    Use ASQR to select an n between 0 and n_max (inclusive).
    """
    performances = []
    for k in range(num_R_samples):
        # agent needs to actually sample R! (as opposed to taking expectation)
        sampled_R = agent.sample_mdp()[0]
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], variance_of_simulated_queries, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        performances.append([estimate_performance_ASQR(agent, sampled_R, first_n_sampled_rewards[n], query_cost) for n in range(n_max+1)])
    #import ipdb; ipdb.set_trace()
    avg_performances = np.array(performances).sum(0)
    return avg_performances.argmax()


def copy_agent_with_different_n(agent, n, query_cost):
    agent_copy = copy.deepcopy(agent)
    agent_copy.query_function = QueryFirstNVisits(query_cost, n)
    return agent_copy

def run_SQR(agent, n_max, env, epLen, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    Use SQR to select an n between 0 and n_max (inclusive).
    """
    performances = []
    for k in range(num_R_samples):
        sampled_R, sampled_P = agent.sample_mdp()
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], variance_of_simulated_queries, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        # make copies for run_finite_tabular_experiment
        env_copy = copy.deepcopy(env)
        # is this even used?
        env_copy.R = {kk:(sampled_R[kk], variance_of_simulated_queries) for kk in sampled_R}
        env_copy.P = sampled_P
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
        performances.append([run_finite_tabular_experiment(copy_agent_with_different_n(agent, n, query_cost), 
                                                           env_copy, f_ext, epLen, sampled_rewards=first_n_sampled_rewards[n])[2]
                                                        for n in range(n_max+1)])
    avg_performances = np.array(performances).sum(0)
    return avg_performances.argmax()



# TESTING THAT IT RUNS!
# COPIED from estimate_optimal_queries
#------------------------------------
import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents
seed = np.random.randint(10000)
#seed = 5
numpy_rng = np.random.RandomState(seed)
from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

grid_width=4
epLen = 2 * grid_width - 1
num_episodes = 100
scaling=.1
prob_zero_reward=.9
#query_cost=.5

nAction = 5
states = range(grid_width**2)

reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))

env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
def makeAgent(n, query_cost):
    query_function = query_functions.QueryFirstNVisits(query_cost, n)
    return finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=env.P, R_true=None, query_function=query_function)

# END: COPIED from estimate_optimal_queries
#------------------------------------

# make sure there are non-zero rewards!
print sum([rr[0] for rr in env.R.values()])


# You might need more samples and more exps (and maybe more episodes) to get significant results
num_R_samples = 20
num_exps = 20
n_max = 4

# FIXME: always choosing 0??

# for query_cost = 0, it should usually chose n = n_max
query_cost = 0
agent = makeAgent(1, query_cost)
print sorted([run_SQR(agent, n_max, env, epLen, query_cost, num_R_samples) for k in range(num_exps)])
print sorted([run_ASQR(agent, n_max, query_cost, num_R_samples) for k in range(num_exps)])

# for small query_cost, it should usually chose 0 < n < n_max
query_cost = .01
agent = makeAgent(1, query_cost)
print sorted([run_SQR(agent, n_max, env, epLen, query_cost, num_R_samples) for k in range(num_exps)])
print sorted([run_ASQR(agent, n_max, query_cost, num_R_samples) for k in range(num_exps)])

# for large query_cost, it should usually chose n = 0
query_cost = 1.
agent = makeAgent(1, query_cost)
print sorted([run_SQR(agent, n_max, env, epLen, query_cost, num_R_samples) for k in range(num_exps)])
print sorted([run_ASQR(agent, n_max, query_cost, num_R_samples) for k in range(num_exps)])









#---------------------------------------------------
# CODE FOR LATER
# Code that does Bayesian updating for a Gaussian (?)
# from: http://engineering.richrelevance.com/bayesian-analysis-of-normal-distributions-with-python/
from numpy import sum, mean, size, sqrt
from scipy.stats import norm, invgamma

def draw_mus_and_sigmas(data,m0,k0,s_sq0,v0,n_samples=10000):
    # number of samples
    N = size(data)
    # find the mean of the data
    the_mean = mean(data) 
    # sum of squared differences between data and mean
    SSD = sum( (data - the_mean)**2 ) 

    # combining the prior with the data - page 79 of Gelman et al.
    # to make sense of this note that 
    # inv-chi-sq(v,s^2) = inv-gamma(v/2,(v*s^2)/2)
    kN = float(k0 + N)
    mN = (k0/kN)*m0 + (N/kN)*the_mean
    vN = v0 + N
    vN_times_s_sqN = v0*s_sq0 + SSD + (N*k0*(m0-the_mean)**2)/kN

    # 1) draw the variances from an inverse gamma 
    # (params: alpha, beta)
    alpha = vN/2
    beta = vN_times_s_sqN/2
    # thanks to wikipedia, we know that:
    # if X ~ inv-gamma(a,1) then b*X ~ inv-gamma(a,b)
    sig_sq_samples = beta*invgamma.rvs(alpha,size=n_samples)

    # 2) draw means from a normal conditioned on the drawn sigmas
    # (params: mean_norm, var_norm)
    mean_norm = mN
    var_norm = sqrt(sig_sq_samples/kN)
    mu_samples = norm.rvs(mean_norm,scale=var_norm,size=n_samples)

    # 3) return the mu_samples and sig_sq_samples
    return mu_samples, sig_sq_samples





