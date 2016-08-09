
def sample_R(agent):
    return agent.sample_mdp()[0]

#sampled_R = sample_R(agent)

def sampled_rewards(agent, queries, sampled_R):
    """ queries[s,a] = # of queries to perform on (s,a) """
    # draw samples from R_hat
    samples = {(s,a) : np.random.gaussian(sampled_R[s,a][0], sampled_R[s,a][1], queries[s,a])}
    return samples

# TODO: we could also try sampling the rewards independently for each value of n 
#        (this would make the comparison btw different ns more stochastic)

def estimate_performance(agent, sampled_rewards, query_cost, sampled_R):
    """ we pass the first n sampled_rewards from the function above"""
    R_hat, P_hat = agent.map_mdp()
    updated_R = {}
    for [s,a] in sampled_rewards:
        mu0, tau0 = R_hat[s,a]
        num_samples = len(sampled_rewards[s,a])
        tau1 = tau0 + self.tau * num_samples
        mu1 = (mu0 * tau0 + reward * self.tau * num_samples) / tau1
        updated_R[s,a] = mu1, tau1
    return max(agent.compute_qVals(updated_R, P_hat)[0]) - query_cost * sum(queries.values())


import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents

seed = 1 
numpy_rng = np.random.RandomState(seed)

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment


# TODO: measure *relative* performance?? 
#   (i.e. fraction of excess available rewards you captured)
#   (or: normalize rewards so that expected rewards are 0 (expectation over what??))
#
#   if we DON'T do this, it seems like we're asking for death (in cake-or-death)
#   mismatched priors... :P :P xP
#   worst-case? (or something btw that and the bayesian thing?)
#   speculative: relationship to norms??? (worst-case is like max norm??)
#   (Depmster-Schafer)


# AGENT
grid_width=4
epLen = 2 * grid_width - 1
num_episodes = 101
scaling=.1
prob_zero_reward=.9
query_cost=1.

k = 1000
iters = k

ns = range(9)

nAction = 5
states = range(grid_width**2)

reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))

env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)

def makeAgent(n):
    query_function = query_functions.QueryFirstNVisits(query_cost, n)
    return finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=env.P, R_true=None, query_function=query_function)

def runexp(env, agent, hasP=True):
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Run the experiment
    global seed
    seed += 1
    # returns: cumReward, cumQueryCost, perf, cumRegret
    return run_finite_tabular_experiment(agent, env, f_ext, num_episodes, seed,
                        recFreq=1000, fileFreq=10000, targetPath='')   

def sample_real_mdp(agent): 
    return gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, *agent.sample_mdp())

def rollout_performance(makeAgent, n): 
    agent = makeAgent(n)
    return runexp(sample_real_mdp(agent), agent)

# returns outcomes for each rollout
# TODO: should use the same R-tilde for each n!!
# TODO: "branching" the agents (so use the same partial histories as much as possible)
def performance_rollouts (makeAgent, ns, iters):
    return np.array([[rollout_performance(makeAgent, n) for i in range(iters)] for n in ns])

def average_performance(makeAgent, ns, iters):
    return np.mean(performance_rollouts(makeAgent, ns, iters), axis=1)

#p = average_performance(makeAgent, ns, 10)
#print p


# DK: below
rollouts = performance_rollouts(makeAgent, ns, iters)
# all the rollouts, in order (use indexing to pull out only those that correspond to a given n)
flattened_rollouts = np.concatenate(rollouts, axis=0)
rollout_performances = [rollout[2] for rollout in flattened_rollouts]
rollout_returns = [rollout[0] for rollout in flattened_rollouts]

def bootstrap(dataset, estimator, num_samples=10000):
    #print dataset
    bootstrap_inds = np.random.randint(0, len(dataset), len(dataset) * num_samples)
    bootstrap_exs = [dataset[ind] for ind in bootstrap_inds]
    return [estimator(bootstrap_exs[samp*len(dataset): (samp + 1)*len(dataset)]) for samp in range(num_samples)]





# TODO: check performances for different values of c
from evaluation import compute_performance
from pylab import *

rollout_performances_c = []
bootstrap_mean_performances_c = []
for cost in [2, 5, 10]:
    rollout_performances_c.append([compute_performance(_return, _performance, cost) for _return, _performance in zip(rollout_returns, rollout_performances)])
    #bootstrap_mean_performances = [bootstrap(rollout_performances[n*iters: (n+1)*iters], lambda x: np.mean(x)) for n in range(len(ns))]
    bootstrap_mean_performances_c.append([bootstrap(rollout_performances_c[-1][n*iters: (n+1)*iters], lambda x: np.mean(x)) for n in range(len(ns))])
    #figure()
    #for i,n in enumerate(ns):
    #    subplot(3,3,i+1)
    #    hist(bootstrap_mean_performances[i], 100)
    #    title("n=" + str(n))
    figure()
    for n in ns[1:-1]:
        hist(bootstrap_mean_performances_c[-1][n], 20)
    








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





