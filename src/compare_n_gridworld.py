import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents

seed = 1 
numpy_rng = np.random.RandomState(seed)

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

# FIXME: swap for loops

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

def bootstrap(dataset, estimator, num_samples=100000):
    #print dataset
    bootstrap_inds = np.random.randint(0, len(dataset), len(dataset) * num_samples)
    bootstrap_exs = [dataset[ind] for ind in bootstrap_inds]
    return [estimator(bootstrap_exs[samp*len(dataset): (samp + 1)*len(dataset)]) for samp in range(num_samples)]





# TODO: check performances for different values of c
from evaluation import compute_performance
from pylab import *

rollout_performances_c = []
bootstrap_mean_performances_c = []
for cost in [.5, 1., 1.5, 2., 2.5]:
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
    
