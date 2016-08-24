import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents

seed = 2 
numpy_rng = np.random.RandomState(seed)

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

# AGENT
grid_width=2
epLen = 2 * grid_width - 1 + 8
num_episodes = 53
scaling=.1
prob_zero_reward=.9
query_cost=1
reward_tau = 100**2

nAction = 5
states = range(grid_width**2)

reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))

reward_probabilities = gridworld.reward_for_action(reward_probabilities, action=0)


env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities, 0)


def modifyPrior(prior): 
    def nonStayKnown(sa, p ): 
        #non 'stay' actions have infinite precision
        _, action = sa
        mu, tau = p

        if action != 0: 
            return (mu, 1e10)
        else: 
            return (mu, tau)


    return { k : nonStayKnown(k, v) for k,v in prior.iteritems() } 

def makeAgent(n):
    def nquery(s, a):
        return (a == 0) * n

    query_function = query_functions.QueryFixedFunction(query_cost, nquery)

    agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=env.P, R_true=None, query_function=query_function, 
                              tau=reward_tau)

    agent.R_prior = modifyPrior(agent.R_prior)
    return agent






def runexp(env, agent, hasP=True):
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Run the experiment
    global seed
    seed += 1
    # returns: cumReward, cumQueryCost, perf, cumRegret
    r = run_finite_tabular_experiment(agent, env, f_ext, num_episodes, seed,
                        recFreq=1000, fileFreq=10000, targetPath='')   

    return r 

def sample_real_mdp(agent): 
    R, P = agent.sample_mdp()
    
    return gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, R, P, 0)

import plot



def rollout_performance(agent, mdp): 
    return runexp(mdp, agent)

import copy
def performance_rollouts (worlds, agents, labels):
    return [(label, w) + rollout_performance(copy.deepcopy(agent), world) + (agent,) 
            for (w, world) in enumerate(worlds) 
            for agent, label in zip(agents, labels)]

ns = [0,1,2,4,8]

def worlds_from_prior(agent, iters):
    return [sample_real_mdp(agent) for i in range(iters)]

worlds = worlds_from_prior(makeAgent(np.inf), 1)
worlds = [
    gridworld.make_gridworld(2, epLen, gridworld.reward_for_action([0,1,0,0], action=0), reward_noise=0),
    gridworld.make_gridworld(2, epLen, gridworld.reward_for_action([0,0,1,0], action=0), reward_noise=0),
    ]*40

agents = map(makeAgent, ns)

obs = performance_rollouts(worlds, agents, ns)


import pandas 
data = pandas.DataFrame(obs, columns='n,R,cumReward,cumQueryCost,perf,cumRegret,_'.split(','))
data.to_csv('sqr3.csv')


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





