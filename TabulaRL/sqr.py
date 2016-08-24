import numpy as np
import gridworld
import query_functions
import finite_tabular_agents
from copy import deepcopy

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment

def fillPrior(env, prior, fill): 
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

