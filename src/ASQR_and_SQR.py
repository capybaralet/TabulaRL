
import numpy
np = numpy
import scipy
import copy

#from experiment import run_finite_tabular_experiment
from TabulaRL.dk_run_finite_tabular_experiment import run_finite_tabular_experiment
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.query_functions import QueryFirstNVisits


variance_of_simulated_queries = 1.

# TODO: maintain a posterior over the variance
# TODO: case where P (i.e. T) is unknown


"""
This ended up not being a very good way of writing this code; it's pretty inefficient, and annoying to pass things back and forth.
I plan to just port everything into dk_exp_script
"""


#def sample_rewards(agent, queries, sampled_R):
#   """ queries[s,a] = # of queries to perform on (s,a) """
#   return {(s,a) : np.random.normal(sampled_R[s,a], 1, queries[s,a]) for (s,a) in agent.P_prior.keys()}

#---------------------------------------------
# ASQR stuff

def estimate_returns_ASQR(agent, sampled_R, sampled_rewards, num_episodes_remaining):
    """
    sampled_R: a dictionary of mean rewards for each (s,a) pair, sampled from the agent's posterior.
    sampled_rewards: a dictionary of (several) sampled rewards for each (s,a) pair, sampled from sampled_R

    returns - ASQR estimate of agent's returns
    """
    _, P_hat = agent.map_mdp()
    R_hat = agent.R_prior
    updated_R = {}
    for [s,a] in sampled_rewards:
        mu0, tau0 = R_hat[s,a]
        num_samples = len(sampled_rewards[s,a])
        tau1 = tau0 + agent.tau * num_samples
        mu1 = (mu0 * tau0 + sum(sampled_rewards[s,a]) * agent.tau) / tau1
        updated_R[s,a] = mu1
    # TODO: shouldn't use P_hat when the environment is unknown (?)
    return agent.compute_qVals_true(updated_R, P_hat, sampled_R, P_hat)

def run_ASQR(agent, ns, num_episodes_remaining, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    Use ASQR to select an n between 0 and n_max (inclusive).

    returns - estimated n_best
    """
    n_max = max(ns)
    max_returns = []
    min_returns = []
    returns = []
    performances = []
    for k in range(num_R_samples):
        sampled_R, sampled_P = agent.sample_mdp() # TODO: shouldn't use P_hat when the environment is unknown (?)
        _, P_hat = agent.map_mdp()
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], variance_of_simulated_queries, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        max_returns.append(agent.compute_qVals(sampled_R, P_hat)[1][0][0])
        min_returns.append(- agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, P_hat)[1][0][0])
        these_returns = [estimate_returns_ASQR(agent, sampled_R, first_n_sampled_rewards[n], num_episodes_remaining)
                                    for n in ns]
        returns.append([num_episodes_remaining * rr[0] for rr in these_returns])
        performances.append([returns[-1][ind] - query_cost * sum([len(qq) for qq in first_n_sampled_rewards[n].values()])
                                    for ind, n in enumerate(ns)])
    avg_performances = np.array(performances).sum(0)
    return avg_performances.argmax(), returns, max_returns, min_returns



#---------------------------------------------
# SQR stuff

def copy_agent_with_different_n(agent, n, query_cost):
    agent_copy = copy.deepcopy(agent)
    agent_copy.query_function = QueryFirstNVisits(query_cost, n)
    return agent_copy

def count_queries_and_estimate_returns_SQR(agent, env, sampled_R, sampled_rewards, num_episodes_remaining):
    """
    Use SQR to run an experiment in a simulated environment with a given agent
    """
    env_copy = copy.deepcopy(env)
    env_copy.R = {kk:(sampled_R[kk], variance_of_simulated_queries) for kk in sampled_R}
    # TODO: case where P is unknown
    #env_copy.P = sampled_P
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
    # we just count the queries (by setting query_cost = 1)
    results = run_finite_tabular_experiment(agent,
                                   env_copy, f_ext, num_episodes_remaining, sampled_rewards=sampled_rewards,
                                   printing=0,
                                   saving=0)
    # returns rewards, num_queries
    return results[0], results[1]

# TODO: return visit counts instead of num_queries
def run_SQR(agent, ns, env, num_episodes_remaining, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    run SQR (for an undefined query cost)
    """
    n_max = max(ns)
    max_returns = []
    min_returns = []
    returns = []
    num_queries = []
    for k in range(num_R_samples):
        sampled_R, sampled_P = agent.sample_mdp()
        _, P_hat = agent.map_mdp()
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], variance_of_simulated_queries, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        max_returns.append(agent.compute_qVals(sampled_R, P_hat)[1][0][0])
        min_returns.append(- agent.compute_qVals({kk: -sampled_R[kk] for kk in sampled_R}, P_hat)[1][0][0])
        agent_copies =[copy_agent_with_different_n(agent, n, query_cost=1) for n in ns]
        these_results = [count_queries_and_estimate_returns_SQR(
                                               agent_copies[ind],
                                               env, 
                                               sampled_R, first_n_sampled_rewards[n], num_episodes_remaining)
                                    for ind, n in enumerate(ns)]
        returns.append([rr[0] for rr in these_results])
        num_queries.append([rr[1] for rr in these_results])
    return num_queries, returns, max_returns, min_returns


