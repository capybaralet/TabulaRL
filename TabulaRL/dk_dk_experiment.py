import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents

seed = np.random.randint(10000) #1
seed = 1
numpy_rng = np.random.RandomState(seed)

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
from environment import TabularMDP
import query_functions

print "finished imports"
targetPath = 'test.csv'


# HYPER
grid_width = 7
nEps = 1000
epLen = 20
query_cost = 1.
prob_zero_reward = .9
scaling = 1 # what is?

print "grid_width=",grid_width
print "query_cost=",query_cost
print "episode_length=",epLen

# states (lexical order)
states = range(grid_width**2)
# reward means
reward_means = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))
print reward_means

#query_function = query_functions.QueryFirstNVisits(query_cost, 4)
#query_function = query_functions.RewardProportional(agent, query_cost, .04)
#query_function = query_functions.EntropyThreshold(agent, query_cost, .5)


max_num_visits = [1,2,4,8,16,32,64]

PSRL_results = []
PSRL_visits = []
for n in max_num_visits:
    print "n=", n

    # Make the environment
    env = gridworld.make_gridworld(grid_width, epLen, reward_means)
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
    # Make the agent
    alg = finite_tabular_agents.PSRL
    agent = alg(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=None,
                              R_true=None)
    # Make the query function
    query_function = query_functions.QueryFirstNVisits(query_cost, n)
    query_function.setEnvAgent(env, agent)

    # Run the experiment
    result = run_finite_tabular_experiment(agent, env, f_ext, nEps, seed,
                        recFreq=1000, fileFreq=10000, targetPath=targetPath, query_function=query_function)
    PSRL_results.append(result)
    PSRL_visits.append(query_function.visit_count)


eGreedy_results = []
eGreedy_visits = []
for n in max_num_visits:
    print "n=", n

    # Make the environment
    env = gridworld.make_gridworld(grid_width, epLen, reward_means)
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
    # Make the agent
    alg = finite_tabular_agents.EpsilonGreedy
    agent = alg(env.nState, env.nAction, env.epLen,
                              scaling=scaling, 
                              P_true=None,
                              R_true=None)
    # Make the query function
    query_function = query_functions.QueryFirstNVisits(query_cost, n)
    query_function.setEnvAgent(env, agent)

    # Run the experiment
    result = run_finite_tabular_experiment(agent, env, f_ext, nEps, seed,
                        recFreq=1000, fileFreq=10000, targetPath=targetPath, query_function=query_function)
    eGreedy_results.append(result)
    eGreedy_visits.append(query_function.visit_count)


from pylab import *
figure()
plot(max_num_visits, [rr[2] for rr in eGreedy_results])
plot(max_num_visits, [rr[2] for rr in PSRL_results])
title('performance: PSRL (green) vs. epsilon-greedy (blue)')
# TODO: save this stuff...
figure()
plot(max_num_visits, [rr[2] for rr in eGreedy_results])
plot(max_num_visits, [rr[2] for rr in PSRL_results])
title('cumRegret: PSRL (green) vs. epsilon-greedy (blue)')


