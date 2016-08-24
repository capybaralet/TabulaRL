import numpy as np
import gridworld
import query_functions
import finite_tabular_agents
from copy import deepcopy

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

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


def runexp(env, agent, seed):
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # returns: cumReward, cumQueryCost, perf, cumRegret
    return run_finite_tabular_experiment(agent, env, f_ext, env.num_episodes, seed,
                        recFreq=1000, fileFreq=10000, targetPath='')   

def sample_real_mdp(agent): 
    R, P = agent.sample_mdp()
    
    world = gridworld.make_mdp(agent.nState, agent.nAction, agent.epLen, R, P, agent.tau**-.5)
    world.num_episodes = agent.env.num_episodes
    return world



def rollout_performance(agent, mdp, seed): 
    return runexp(mdp, agent, seed)

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

    print "perf shape", performance.shape
    return np.mean(performance, axis=1)


def genPrior(env, prior, fill): 
    fillPrior = { (s,a) : fill for s in range(env.nState) for a in range(env.nAction) }
    fillPrior.update(**prior)

    return fillPrior


def MakePSRLLimitedQueryAgent(reward_tau, env, prior, query_function):
    agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
              scaling=.1, 
              P_true=env.P, R_true=None, query_function=deepcopy(query_function), 
              tau=reward_tau)

    agent.env = env
    agent.R_prior = deepcopy(prior)
    return agent





def MakeSQRAgent(reward_tau, env, prior, query_functions, iters): 
    def makeThisAgent(query_function):
        return MakePSRLLimitedQueryAgent(reward_tau, env, prior, query_function)


    prototypicalAgent = makeThisAgent(query_functions[-1])
    worlds = worlds_from_prior(prototypicalAgent, iters)
    agents = [makeThisAgent(qfunc) for qfunc in query_functions]

    perfs = compute_average_performance(worlds, agents)

    best = np.argmax(perfs)
    return makeThisAgent(query_functions[best])

