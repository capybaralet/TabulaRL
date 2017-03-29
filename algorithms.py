"""
WIP

This script should contain all of the RL algorithms for this repo.

So far, my implementations follow Sutton-Barto 2017.


I index as:
    P   s, a, s'
    R   s, a
    pi  s, a
"""

import numpy as np
#from utilities import sample


# TODO: I still haven't implemented a (generic) off-policy algorithm for Q-values...
#   ...but I have the iterative policy evaluation, that's enough for now!



# TODO: pass random seeds for reproducibility!


# N.B. this is IN-PLACE version!!
# TODO: Q-learning version (??)
def iterative_policy_evaluation(pi, P, R, gamma=1, tol=1e-10):
    # Initialize an array V (s) = 0, for all s in S+
    V = np.zeros(P.shape[-1])
    delta = np.inf
    num_iterations = 0
    while delta > tol:
        num_iterations += 1
        delta = 0 # max (error of V) over all states
        for s in range(P.shape[0]): # don't update terminal state
            v = V[s]
            # the sum over s' is implicit; maybe we can do that for the sum over a, as well, for a more efficient implementation
            inner_sum = np.array([ np.sum( P[s,a] * ( R[s,a] + gamma * V )) for a in range(len(pi[s])) ])
            V[s] = np.sum(pi[s] * inner_sum)
            delta = max(delta, np.abs(v - V[s]))
        #print delta, V
    return V, num_iterations


# in the book, they only have this for greedy pi; does it work for arbitrary pi??
# ^ I think that requires importance sampling
def off_policy_every_visit_MC_control(mdp, mu=None, pi='greedy', gamma=1, tol=1e-10, num_iterations=1e6, Q=None):
    # TODO: allow stochastic policies
    # TODO: deal with discounting somehow??
    """
    TODO: P_s is the distribution over start states
    """

    R = mdp.R
    nS, nA = R.shape
    P = mdp.P
    try:
        P_s = mdp.P_s
    except:
        P_s = np.array([1] + [0,] * (nS-1))

    if Q is None:
        Q = np.zeros((nS, nA))
    C = np.zeros((nS, nA))
    if pi == 'greedy':
        greedy = True
        pi = np.argmax(Q, axis=1) # greedification
    else: # evaluate a fixed target policy (I hope that works...)
        greedy = False
    if mu is None: # totally random
        mu = 1./nA * np.ones((nS, nA))
    delta = np.inf
    for traj in range(int(num_iterations)):#while delta > tol:
        # generate a trajectory
        terminal = 0
        s = np.argmax(np.random.multinomial(1, P_s))
        states = [s]
        actions = []
        rewards = []
        while not terminal:
            a = np.argmax(np.random.multinomial(1, mu[s]))
            actions.append(a)
            # TODO: change the order of s,r in the environments
            s, r, terminal = mdp.step(s,a)
            print s, r, terminal
            rewards.append(r)
            states.append(s)
            # TODO: this is a hack!
            terminal = (mdp.terminal == s)
        T = len(actions)
        # END generate a trajectory

        #delta = 0 # max (error of Q) over all states

        G = 0
        W = 1
        for t in range(T)[::-1]:
            s = states[t]
            a = actions[t]
            r = rewards[t]
            #import ipdb; ipdb.set_trace()
            G = gamma * G + r
            C[s,a] += W
            Q[s,a] += W / C[s,a] * (G - Q[s,a])
            #delta = max(delta, np.abs(v - V[s]))
            if greedy:
                pi[s] = np.argmax(Q[s]) # greedification
            if a != pi[s]:
                print "break"
                break
            W *= 1./mu[s,a]
    
    return Q



MCC = off_policy_every_visit_MC_control

