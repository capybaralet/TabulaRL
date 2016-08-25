import numpy as np
import gridworld
from copy import deepcopy

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment


def rollout_performance(agent, env, seed):
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # returns: cumReward, cumQueryCost, perf, cumRegret
    return run_finite_tabular_experiment(agent, env, f_ext, env.num_episodes, seed,
                        recFreq=1000, fileFreq=10000, targetPath='')   

def sample_real_mdp(agent): 
    R, P = agent.sample_mdp()
    
    world = gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, R, P, agent.tau**-.5)
    world.num_episodes = agent.env.num_episodes
    return world

def worlds_from_prior(agent, iters):
    return [sample_real_mdp(agent) for i in range(iters)]

def performance_rollouts(worlds, agents, labels):

    stuff = [ (w, world, deepcopy(agent), label) 
                for w, world in enumerate(worlds) 
                for agent, label in zip(agents, labels)
            ]


    return [ (label, w) + rollout_performance(agent, world, i) + (agent,) for 
            i, (w, world, agent, label) in enumerate(stuff)
            ]

def compute_average_performance(worlds, agents):
    performance =  np.array([ 
            [rollout_performance(deepcopy(agent), world, seed)[2] for seed, world in enumerate(worlds) ]
            for agent in agents
            ])

    return np.mean(performance, axis=1)




