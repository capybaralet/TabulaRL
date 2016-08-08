import numpy as np
import argparse
import gridworld
import query_functions
import finite_tabular_agents
from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment
from environment import make_stochasticChain
seed = 1
seed = np.random.randint(10000) #1
#numpy_rng = np.random.RandomState(seed)
np.random.seed(seed)

import time
import os
filename = os.path.basename(__file__)
save_dir = 'results__' + filename
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#from name_print import make_save_str


# Note that for the chain, our model is incorrect: the intermediate states have deterministic rewards!
# TODO: make better logging / monitoring
# TODO: other environments?
# TODO: use job launchers/schedulers!
# Separately:
#   run a job
#   schedule many jobs
#   analyze results
#   ... but have all of these things in one place somehow??


"""
So far, we're just tuning n once (at the beginning), using SQR

First steps:
    check the sensitivity of n_best to randomness

We need to run enough experiments to see which n works best (in the real environment)
    It seems pretty random, and depends on the query cost

"""

# environment:
if 0: # grid-world
    grid_width=8
    epLen = 2 * grid_width - 1
    prob_zero_reward=.9
    nAction = 5
    states = range(grid_width**2)
    #reward_probabilities = numpy_rng.binomial(1, 1 - prob_zero_reward, len(states)) * numpy_rng.uniform(0, 1, len(states))
    reward_probabilities = np.random.binomial(1, 1 - prob_zero_reward, len(states)) * np.random.uniform(0, 1, len(states))
    env = gridworld.make_gridworld(grid_width, epLen, reward_probabilities)
elif 1: # chain
    chain_len = 3
    epLen = chain_len
    env = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-chain_len)
f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

# AGENT
alg = finite_tabular_agents.PSRLLimitedQuery
initial_agent = alg(env.nState, env.nAction, env.epLen, P_true=None, R_true=None)

# experiment hparams:
num_episodes = 20000
num_R_samples = 100
query_cost = 100. # TODO: grid search
Ns = [0,1,3,10,30,100]
#Ns = range(100)
#Ns = [num_episodes * env.nState * env.nAction] # always query


# Here, we compare our way of tuning n (running multiple trials in a sampled environment)
#  with the best possible way (actually running multiple trials in the real environment)
if 1:
    print "\n running comparison with ground truth experiments..."
    all_save_strs = []
    mean_perfs = {}
    all_perfs = {}
    SQR_mean_perfs = {}
    SQR_all_perfs = {}
    all_visit_counts = {}
    SQR_all_visit_counts = {}
    # TODO: switch the order of these for loops (so that we compare different n on the same R)
    for n in Ns:
        perfs = []
        SQR_perfs = []
        visit_counts = []
        R_priors = []
        SQR_visit_counts = []

        for use_real_env in [0, 1]:
            for k in range(num_R_samples):
                #for n in Ns:
                    t1 = time.time()
                    #save_str = make_save_str([filename, n, use_real_env, Rsample])
                    save_str = save_dir + '/query_cost=' + str(query_cost) + '__n=' + str(n) + '__use_real_env=' + str(use_real_env) + '__k=' + str(k)
                    all_save_strs.append(save_str)
                    print save_str

                    if use_real_env:
                        env_sample = env
                    else:
                        sampled_R = initial_agent.sample_mdp()[0]
                        print sampled_R
                        env_sample = gridworld.make_mdp(env.nState, env.nAction, env.epLen, sampled_R, env.P)

                    query_function = query_functions.QueryFirstNVisits(query_cost, n)
                    agent = alg(env.nState, env.nAction, env.epLen,
                                              P_true=None, R_true=None,
                                              query_function=query_function)
                    query_function.setAgent(agent)

                    # Run the experiment
                    # returns:  cumReward, cumQueryCost, perf, cumRegret 
                    result = run_finite_tabular_experiment(agent, env_sample, f_ext, num_episodes, seed,
                                        recFreq=1000, fileFreq=10000, targetPath=save_str, query_function=query_function,
                                        printing=1)
                    if use_real_env:
                        perfs.append(result[2]) 
                        R_priors.append(agent.R_prior)
                        visit_counts.append(query_function.visit_count)
                    else:
                        SQR_perfs.append(result[2]) 
                        SQR_visit_counts.append(query_function.visit_count)
                    
                    print time.time() - t1

        
        all_perfs[n] = perfs
        all_visit_counts[n] = visit_counts
        mean_perfs[n] = np.mean(perfs)
        print mean_perfs
        SQR_all_visit_counts[n] = SQR_visit_counts
        SQR_all_perfs[n] = SQR_perfs
        SQR_mean_perfs[n] = np.mean(SQR_perfs)
        print SQR_mean_perfs

    if 1: # plot regret
        from plot_regret import plot_regret
        [plot_regret(save_str) for save_str in all_save_strs]

    SQR_n_best = max(SQR_mean_perfs, key=SQR_mean_perfs.get)
    print "SQR_n_best =", SQR_n_best
    true_n_best = max(mean_perfs, key=mean_perfs.get)
    print "true_n_best =", true_n_best
            
    #print "blue is SQR"        



