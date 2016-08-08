"""

To start with, we assume that T is known (probably it's simple to extend to unknown T)


We do the following (k times):
    Sample a reward function, R ~ P(R)
        Optional: estimate optimal returns via planning with R
    Sample N queries from R for every (s,a) pair
        For every n in {1, ..., N}:
            Compute a new posterior P(R | first n sampled queries)
            Estimate optimal returns via planning over the new posterior
            
    We'd like to have a way to make a different number of queries to different (s,a).
        But naively, there are exponentially many different combinations.
            If we used the linear-bandit type approach, then that might tell use which (s,a) we'd want to query at each tstep
            so then we can run the linear bandit thing for different numbers of tsteps, and just search over the number of tsteps!

^ We can put this whole thing "in the loop" and use it to update our query strategy online!

TODO: 
    * implement everything
    * Compare to the thing where we actually run PSRL on the sampled R to compare the different values of n
    * Evaluate sensitivity to all random elements in the algo:
        ** Sampling R
        ** Sampling queries from R
        ** Running PSRL



can we use Rmax as a heuristic for what N should be??
    i.e. we'd like to turn the query_cost and gamma into epsilon and delta (for the PAC guarantees of Rmax), and then chose N based on that.

"""


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

# AGENT
grid_width=8
epLen = 15
scaling=.1
prob_zero_reward=.9
query_cost=.5

nAction = 5
states = range(grid_width**2)

reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))

env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
def makeAgent(n):
    query_function = query_functions.QueryFirstNVisits(query_cost, n)
    return finite_tabular_agents.PSRL(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=env.P, R_true=False, query_function=query_function)

def runexp(env, agent, hasP=True):
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    P_true =env.P
    if not hasP: 
        P_true=False

    # Run the experiment
    global seed
    seed += 1
    return run_finite_tabular_experiment(agent, env, f_ext, 1, seed,
                        recFreq=1000, fileFreq=10000, targetPath='')   

def sample_real_mdp(agent): 
    return gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, *agent.sample_mdp())

def rollout_performance(makeAgent, n): 
    agent = makeAgent(n)
    return runexp(sample_real_mdp(agent), agent)


def performance_rollouts (makeAgent, ns, iters):
    return np.array([[rollout_performance(makeAgent, n) for i in range(iters)] for n in ns])

def average_performance(makeAgent, ns, iters):
    return np.mean(performance_rollouts(makeAgent, ns, iters), axis=1)


p = average_performance(makeAgent, range(0,4), 10)
print p





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





