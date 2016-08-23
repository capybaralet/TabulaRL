'''
Script to run simple tabular RL experiments.

author: iosband@stanford.edu


Modified by DK to be more suited to our experiment settings (for running and logging lots of experiments)
    clean-up code
    change monitoring
'''
import numpy as np
import pandas as pd
import cPickle as pickle
from shutil import copyfile

# TODO: currently assumes that agent.query_function.visit_count exists (i.e. that we're using FirstNVisits to query)
def run_finite_tabular_experiment(agent, env, f_ext, nEps, seed=1,
                    recFreq=100, fileFreq=1000, targetPath='tmp',
                    sampled_rewards=None,
                    printing=False,
                    saving=True):
    data = []
    visit_counts = {}
    qVals, qMax = env.compute_qVals()
    #np.random.seed(seed)

    # We'll track the cumulative regret (including the query cost).
    cumRegretIncludingQueryCost = 0
    cumQueryCost = 0
    cumReward = 0

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

        cumRegretIncludingQueryCost += epRegret
        cumRegretIncludingQueryCost += epQueryCost
        cumQueryCost += epQueryCost
        cumReward += epReward

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
            data.append([ep, cumRegretIncludingQueryCost, cumQueryCost, cumReward])
            if printing:
                print 'episode:', ep, 'epReward:', epReward, 'epQueryCost:', epQueryCost, 'perf:', perf, 'cumRegretIncludingQueryCost:', cumRegretIncludingQueryCost, 'cumQueryCost:', cumQueryCost

        if saving and (ep % max(fileFreq, recFreq) == 0):
            dt = pd.DataFrame(data,
                              columns=['episode', 'cumRegretIncludingQueryCost', 'cumQueryCost', 'cumReward'])
            print 'Writing to file ' + targetPath + '.csv'
            dt.to_csv('tmp.csv', index=False, float_format='%.2f')
            copyfile('tmp.csv', targetPath + '_.csv')

            # log visit counts in a pkl
            visit_counts[ep] = dict(agent.query_function.visit_count)
            filepath = targetPath + '__visit_counts.pkl'
            with open(filepath, 'wb') as file_:
                pickle.dump(visit_counts, file_)

    return cumReward, cumQueryCost, perf, cumRegretIncludingQueryCost

