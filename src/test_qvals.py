import numpy as np
import argparse
import itertools
import copy
from collections import defaultdict
from pylab import *

import TabulaRL.gridworld as gridworld
import TabulaRL.query_functions as query_functions
import TabulaRL.finite_tabular_agents as finite_tabular_agents
from TabulaRL.feature_extractor import FeatureTrueState
from TabulaRL.environment import make_stochasticChain, make_deterministicChain

from generic_functions import add_dicts, dict_argmax, is_power2, sample_gaussian, update_gaussian_posterior_mean, pysave

reward_noise = .5
query_cost = 1




# FIXME: ret_true == 0???
# compute actual policy...





def get_env(enviro):
    if enviro.startswith('grid'):
        grid_width = int(enviro.split('grid')[1])
        epLen = 2 * grid_width - 1
        reward_means = np.diag(np.linspace(0, 1, grid_width)).flatten()
        envv = gridworld.make_gridworld(grid_width, epLen, gridworld.make_sa_rewards(reward_means, actions = range(5)),
                                         multi_chain=False, gotta_move=True, reward_noise=reward_noise)
    # TODO: leaving the extra states around messes with ASQR!
    elif enviro.startswith('multi_chain'):
        grid_width = int(enviro.split('multi_chain')[1])
        epLen = 2 * grid_width - 1 # agent needs one extra move (only gets reward in a state if it acts in it!)
        reward_means = np.diag(np.linspace(0, 1, grid_width)).flatten()
        envv = gridworld.make_gridworld(grid_width, epLen, gridworld.make_sa_rewards(reward_means, actions = range(5)), 
                                         multi_chain=True, gotta_move=True)
    elif enviro.startswith('det_chain'):
        chain_len = int(enviro.split('det_chain')[1])
        epLen = chain_len
        envv = make_deterministicChain(chain_len, chain_len)
    elif enviro.startswith('stoch_chain'):
        chain_len = int(enviro.split('stoch_chain')[1])
        epLen = chain_len
        envv = make_stochasticChain(chain_len, max_reward=((chain_len - 1.)/chain_len)**-(chain_len-1))
    elif enviro.startswith('longY'):
        nState = int(enviro.split('longY')[1])
        epLen = nState - 1
        reward_means = np.zeros(nState)
        reward_means[-1] = 1
        envv = gridworld.make_longY(nState, epLen, 
                                     rewards=gridworld.make_sa_rewards(reward_means, actions=range(2)), reward_noise=reward_noise)
    else:
        assert False
    f_ext = FeatureTrueState(envv.epLen, envv.nState, envv.nAction, envv.nState)
    return envv, f_ext


def qvals_array(qvals_dict, envv):
    rval = np.empty((envv.nState, envv.epLen, envv.nAction))
    for kk in qvals_dict:
        print kk, len(qvals_dict[kk])
        rval[kk] = qvals_dict[kk]
    return rval

def sorted_items(dd):
    return [ite[1] for ite in iter(sorted(dd.iteritems()))]

def moving_average(values,window):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'valid')
    return np.hstack((smas[0]*np.ones(window-1), smas)) # as a numpy array

def compute_qVals_arr(self, R, P, R_true, P_true):
    '''
    Evaluate an agent's expected returns when it plans according to R and P
    in an environment defined by R_true, P_true

    Returns:
        The true expected returns of the agent, 
         what it thinks its expected returns are.
    '''
    qVals = -99*np.ones((self.nState, self.epLen, self.nAction))
    qVals_true = -99*np.ones((self.nState, self.epLen, self.nAction))
    qMax = -99*np.ones((self.epLen+1, self.nState))
    qMax_true = -99*np.ones((self.epLen+1, self.nState))

    qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
    qMax_true[self.epLen] = np.zeros(self.nState, dtype=np.float32)
    
    for i in range(self.epLen):
        j = self.epLen - i - 1
        qMax[j] = np.zeros(self.nState, dtype=np.float32)
        qMax_true[j] = np.zeros(self.nState, dtype=np.float32)
         
        for s in range(self.nState):
            qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
            qVals_true[s, j] = np.zeros(self.nAction, dtype=np.float32)

            for a in range(self.nAction):
                qVals[s, j, a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])
                qVals_true[s, j, a] = R_true[s, a] + np.dot(P_true[s, a], qMax_true[j + 1])
    
            # agent acts according to what it believes
            a = np.argmax(qVals[s, j])
            print a
            # we compute both its estimate of the value of this state/tstep, and the true value
            qMax[j, s] = qVals[s, j, a]
            qMax_true[j, s] = qVals_true[s, j, a]
    
        print "np.round(qVals_true)"
        print np.round(qVals_true[:,j])
        print "np.round(qVals) "
        print np.round(qVals[:,j]) 
        print "np.round(qMax_true)"
        print np.round(qMax_true[j])
        print "np.round(qMax)"
        print np.round(qMax[j])
        print ''
        #import ipdb; ipdb.set_trace()
    # M_true, M_prior
    #import ipdb; ipdb.set_trace()
    return qVals_true, qVals, qMax_true, qMax

def compute_qVals_true__(self, R, P, R_true, P_true):
    '''
    Evaluate an agent's expected returns when it plans according to R and P
    in an environment defined by R_true, P_true

    Returns:
        The true expected returns of the agent, 
         what it thinks its expected returns are.
    '''
    qVals = {}
    qMax = {} # aka "V"
    qVals_true = {}
    qMax_true = {}

    qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
    qMax_true[self.epLen] = np.zeros(self.nState, dtype=np.float32)
    
    for i in range(self.epLen):
        j = self.epLen - i - 1
        qMax[j] = np.zeros(self.nState, dtype=np.float32)
        qMax_true[j] = np.zeros(self.nState, dtype=np.float32)
         
        for s in range(self.nState):
            qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
            qVals_true[s, j] = np.zeros(self.nAction, dtype=np.float32)

            for a in range(self.nAction):
                qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])
                qVals_true[s, j][a] = R_true[s, a] + np.dot(P_true[s, a], qMax_true[j + 1])
    
            # agent acts according to what it believes
            a = np.argmax(qVals[s, j])
            # we compute both its estimate of the value of this state/tstep, and the true value
            qMax[j][s] = qVals[s, j][a]
            qMax_true[j][s] = qVals_true[s, j][a]
    
        #import ipdb; ipdb.set_trace()
    # M_true, M_prior
    #import ipdb; ipdb.set_trace()
    return qVals_true, qVals, qMax_true[0][0], qMax[0][0]

def compute_qVals(self, R, P):
    qVals = {}
    qMax = {}
    qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
    for i in range(self.epLen):
        j = self.epLen - i - 1
        qMax[j] = np.zeros(self.nState, dtype=np.float32)
        for s in range(self.nState):
            qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
            for a in range(self.nAction):
                qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])
            a = np.argmax(qVals[s,j])
            print "a", a
            qMax[j][s] = np.max(qVals[s, j])
            print qMax[j][s], qVals[s,j][a]
    return qVals, qMax


# debugging
if 1:
    ret_true_, ret_belief_, mt_, m_ = compute_qVals_true__(agent, est_R, est_P, R_means, envv.P)
    ret_true_ = qvals_array(ret_true_, envv)
    ret_belief_ = qvals_array(ret_belief_, envv)

    #qvt, qv, vt, v = compute_qVals_arr(agent, est_R, est_P, R_means, envv.P)
    qvt, qv, vt, v = compute_qVals_arr(agent, est_R, envv.P, R_means, envv.P)
    qvv, vv = compute_qVals(agent, R_means, envv.P)
    assert False


enviros = ['grid2']#, 'det_chain3', 'multi_chain3', 'longY4']

# RUN TESTS
for enviro in enviros:
    envv, f_ext = get_env(enviro)
    R_means = {sa: envv.R[sa][0] for sa in envv.R}

    # AGENT
    alg = finite_tabular_agents.PSRLLimitedQuery
    if enviro.startswith('grid') or enviro.startswith('multi_chain') or enviro.startswith('longY'):
        initial_agent = alg(envv.nState, envv.nAction, envv.epLen, 
                             P_true=envv.P, R_true=None, reward_depends_on_action=False, tau=1/reward_noise**2, tau0=1/reward_noise**2)
    else:
        initial_agent = alg(envv.nState, envv.nAction, envv.epLen, 
                             P_true=envv.P, R_true=None, tau=1/reward_noise**2, tau0=1/reward_noise**2)
    query_function = query_functions.AlwaysQuery(query_cost)
    query_function.setAgent(initial_agent)
    agent = initial_agent

    # -------------------------------------------------------
    # TESTS:
    print "\n testing enviro=", enviro
    # before learning:
    est_R, est_P = agent.map_mdp()
    ret_true, ret_belief = agent.compute_qVals_true(est_R, est_P, R_means, envv.P)
    ret_belief__ = agent.compute_qVals(est_R, est_P)[1][0][0]
    ret_opt, ret_opt_ = agent.compute_qVals_true(R_means, envv.P, R_means, envv.P)
    ret_opt__ = agent.compute_qVals(R_means, envv.P)[1][0][0]
    # check compute_qVals_true and compute_qVals are consistent
    assert ret_opt == ret_opt_ == ret_opt__
    assert ret_belief == ret_belief__
    print ret_opt, ret_belief, ret_true

    # S,T,A
    qvals_opt, _ = agent.compute_qVals(R_means, envv.P)
    qvals_belief, _ = agent.compute_qVals(est_R, est_P)
    qvals_opt = qvals_array(qvals_opt, envv)[:,0]
    qvals_belief = qvals_array(qvals_belief, envv)[:,0]
    print np.sum((qvals_opt - qvals_belief)**2)

    #assert False
    #------------------
    # perform learning:
    cumReward = 0
    cumQueryCost = 0 
    epRewards = []
    for ep in xrange(1, 1001): 
        epReward = 0
        envv.reset() # return to tstep=state=0
        agent.update_policy(ep)
        pContinue = 1
        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(envv)

            action = agent.pick_action(oldState, h)
            query, queryCost = agent.query_function(oldState, action, ep, h)
            cumQueryCost += queryCost

            reward, newState, pContinue = envv.advance(action)
            cumReward += reward 
            epReward += reward 
            agent.update_obs(oldState, action, reward, newState, pContinue, h, query)
        epRewards.append(epReward)

    #------------------
    # after learning:
    print "done learning"
    est_R, est_P = agent.map_mdp()
    ret_true, ret_belief = agent.compute_qVals_true(est_R, est_P, R_means, envv.P)
    ret_belief__ = agent.compute_qVals(est_R, est_P)[1][0][0]
    ret_opt, ret_opt_ = agent.compute_qVals_true(R_means, envv.P, R_means, envv.P)
    ret_opt__ = agent.compute_qVals(R_means, envv.P)[1][0][0]
    # check compute_qVals_true and compute_qVals are consistent
    assert ret_opt == ret_opt_ == ret_opt__
    assert ret_belief == ret_belief__
    # check that the agent has learned the correct returns
    # FIXME: ret_true == 0???
    print ret_opt, ret_belief, ret_true

    # check that the agent has learned the correct Qvalues (mostly...)
    qvals_opt, _ = agent.compute_qVals(R_means, envv.P)
    qvals_belief, _ = agent.compute_qVals(est_R, est_P)
    qvals_opt = qvals_array(qvals_opt, envv)[:,0]
    qvals_belief = qvals_array(qvals_belief, envv)[:,0]
    # What matters here is that the Qvalues are correct at the relevant time-steps... hard to evaluate...
    print np.sum((qvals_opt - qvals_belief)**2)

    # TODO: plot qvals better
    figure()
    suptitle(enviro)
    subplot(221)
    title("qvals_opt")
    imshow(qvals_opt, cmap="Greys", interpolation='none')
    subplot(222)
    title("qvals_belief")
    imshow(qvals_belief, cmap="Greys", interpolation='none')
    subplot(223)
    title("reward function")
    plot(sorted_items(R_means)[1::envv.nAction], label="R_true")
    plot(sorted_items(est_R)[1::envv.nAction], label="R_est")
    legend(loc=4)
    subplot(224)
    title("rewards")
    plot(len(epRewards) * [ret_opt,], label="max Returns")
    plot(moving_average(epRewards, 10), label="episode Returns")
    legend(loc=4)


    


