'''
Script to run simple tabular RL experiments.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd

from shutil import copyfile

from query_functions import AlwaysQuery

# TODO: make this a default
from feature_extractor import FeatureTrueState


# TODO: currently assumes that agent.query_function.visit_count exists (i.e. that we're using FirstNVisits to query)
def run_finite_tabular_experiment(agent, env, f_ext, nEps, seed=1,
                    recFreq=100, fileFreq=1000, targetPath='tmp.csv',
                    sampled_rewards=None,
                    printing=True):
    '''
    A simple script to run a finite tabular MDP experiment

    Args:
        agent - finite tabular agent
        env - finite TabularMDP
        f_ext - trivial FeatureTrueState
        nEps - number of episodes to run
        seed - numpy random seed
        recFreq - how many episodes between logging
        fileFreq - how many episodes between writing file
        targetPath - where to write the csv

    Returns:
        NULL - data is output to targetPath as csv file
    '''
    data = []
    qVals, qMax = env.compute_qVals()
    #np.random.seed(seed)

    cumRegret = 0
    cumQueryCost = 0
    cumReward = 0
    empRegret = 0

    for ep in xrange(1, nEps + 2):
        # Reset the environment
        env.reset()
        epMaxVal = qMax[env.timestep][env.state]

        agent.update_policy(ep)

        epReward = 0
        epQueryCost = 0 
        epRegret = 0
        pContinue = 1

        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)

            action = agent.pick_action(oldState, h)
            query, queryCost = agent.query_function(oldState, action, ep, h)
            epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]
            epQueryCost += queryCost

            reward, newState, pContinue = env.advance(action)
            if query and sampled_rewards is not None:
                reward = sampled_rewards[oldState, action][agent.query_function.visit_count[oldState, action] - 1]
            epReward += reward 
            agent.update_obs(oldState, action, reward, newState, pContinue, h, query)

        cumReward += epReward
        cumQueryCost += epQueryCost
        cumRegret += epRegret
        empRegret += (epMaxVal - epReward)

        # Variable granularity
        if ep < 1e2:
            recFreq = 10
        elif ep < 1e3:
            recFreq = 100
        elif ep < 1e4:
            recFreq = 1000
        elif ep < 1e5:
            recFreq = 10000
        else:
            recFreq = 100000

        # TODO: clean-up logging (print / save the same stuff!)
        # Logging to dataframe
        perf = cumReward - cumQueryCost

        if ep % recFreq == 0:
            data.append([ep, epReward, cumReward, cumRegret, empRegret, perf])
            if printing:
                print 'episode:', ep, 'epReward:', epReward, 'epQueryCost:', epQueryCost, 'perf:', perf, 'cumRegret:', cumRegret

        if ep % max(fileFreq, recFreq) == 0:
            dt = pd.DataFrame(data,
                              columns=['episode', 'epReward', 'cumReward',
                                       'cumRegret', 'empRegret', 'perf'])
            print 'Writing to file ' + targetPath
            dt.to_csv('tmp.csv', index=False, float_format='%.2f')
            copyfile('tmp.csv', targetPath)

    return cumReward, cumQueryCost, perf, cumRegret


