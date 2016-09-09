import numpy as np
import argparse
import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents

seed = 1 
numpy_rng = np.random.RandomState(seed)

from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.experiment import run_finite_tabular_experiment

# FIXME: swap for loops

# AGENT
grid_width=3
epLen = 2 * grid_width - 1
num_episodes = 11
scaling=.1
prob_zero_reward=.9
query_cost=1.

iters = 1

nAction = 5
states = range(grid_width**2)
reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))
env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)

def makeAgent(query_function):
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
    return gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, *agent.sample_mdp(), reward_noise=agent.tau)

def rollout_performance(makeAgent, qfn): 
    agent = makeAgent(qfn)
    return runexp(sample_real_mdp(agent), agent)

# returns outcomes for each rollout
# TODO: should use the same R-tilde for each n!!
# TODO: "branching" the agents (so use the same partial histories as much as possible)
def performance_rollouts (makeAgent, iters, qfn):
    return np.array([rollout_performance(makeAgent, qfn) for i in range(iters)])

def average_performance(makeAgent, iters, qfn):
    return np.mean(performance_rollouts(makeAgent, iters, qfn))


# TEST QFNS
n = 3
query_fns = []
query_function = query_functions.AlwaysQuery(query_cost)
query_fns.append(query_function)
query_function = query_functions.QueryFirstN(query_cost, n)
query_fns.append(query_function)
query_function = query_functions.QueryFirstNVisits(query_cost, n)
query_fns.append(query_function)
query_function = query_functions.QueryFixedFunction(query_cost, func=lambda s,a: s + a)
query_fns.append(query_function)
query_function = query_functions.DecayQueryProbability(query_cost, func=lambda ep, ts: 1 /(ep+ts+1) )
query_fns.append(query_function)

for qfn in query_fns:
    p = average_performance(makeAgent, 10, qfn)
    print p





