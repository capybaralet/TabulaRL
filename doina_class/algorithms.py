"""
WIP
This script should contain all of the RL algorithms for this repo.
So far, my implementations closely follow Sutton-Barto 2017.

-------------------------------------------------------------------------
These algorithms are all meant to work with the new environment interface (but not necessarily the new POLICY interface)

"""

# TODO: I can seperate these into V-algos and Q-algos (and more...)
# TODO: I can also make them do just one episode of the algorithm by default



import numpy as np

from utilities import sample

# --------------------------------------------------------------------------
# Utilities

def P_pi(P, pi):
    """ P_pi(s' | s) """
    return np.sum(P * pi, axis = 1)

def get_Q(env, V):
    """ compute Q given V """
    P, R, gamma = env.P, env.R, env.gamma
    Q = np.copy(R).astype('float64')
    Q += gamma * np.sum(P * V.reshape((1,1,-1)), axis=-1)
    return Q

# --------------------------------------------------------------------------
# SECTION 4

# section 4.1
def iterative_policy_evaluation(pi, env, tol=1e-10, return_Q=False):
    """ 
    Compute the value function of a given policy in a given environment.

    (in place version)
    """
    P, R, gamma = env.P, env.R, env.gamma
    # Initialize an array V (s) = 0, for all s in S+
    V = np.zeros(P.shape[-1])
    delta = np.inf
    num_iterations = 0
    while delta > tol:
        num_iterations += 1
        # max_s(change in V[s])
        delta = 0
        for s in range(P.shape[0]): # don't update terminal state
            v = V[s]
            # the sum over s' is implicit; maybe we can do that for the sum over a, as well, for a more efficient implementation
            inner_sum = np.array([ np.sum( P[s,a] * ( R[s,a] + gamma * V )) for a in range(len(pi[s])) ])
            V[s] = np.sum(pi[s] * inner_sum)
            delta = max(delta, np.abs(v - V[s]))
    if return_Q:
        return get_Q(env, V)
    return V

# section 4.4
def value_iteration(env, tol=1e-10):
    """ 
    Find the optimal policy and corresponding value function of an environment.
    """
    #V = np.zeros(env.nS + 1)
    V = np.zeros(env.P.shape[-1])
    delta = np.inf
    while delta > tol:
        delta = 0 
        for s in range(env.nS):
            v = V[s]
            V[s] = max([ np.sum( env.P[s,a] * ( env.R[s,a] + env.gamma * V )) for a in range(env.nA)])
            delta = max(delta, np.abs(v - V[s]))
    # compute the greedy policy
    pi = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        pi[s] = np.argmax([ np.sum( env.P[s,a] * ( env.R[s,a] + env.gamma * V )) for a in range(env.nA)])
    return pi, V

    


# --------------------------------------------------------------------------
# SECTION 5

# N.B. We can pass Q so that this can be called iteratively (TODO: do this for Q-sigma as well!)
def off_policy_every_visit_MC(env, mu=None, pi='greedy', tol=1e-10, num_episodes=1e6, C=None, Q=None, stochastic_termination=0):
    """
    mu: any soft policy
    pi: any policy, or "greedy" for control
    """

    R = env.R
    nS, nA = R.shape
    P = env.P
    P_s = env.P_s
    gamma = env.gamma

    # gamma = P_continue * discount
    P_continue = 1 - stochastic_termination * (1 - gamma)
    discount = gamma / P_continue

    if Q is None:
        Q = np.zeros((nS, nA))
        assert C is None
        C = np.zeros((nS, nA))
    if pi == 'greedy':
        greedy = True
        pi = np.argmax(Q, axis=1) # greedification
    else: # evaluate a fixed target policy
        greedy = False
    if mu is None: # totally random
        mu = 1./nA * np.ones((nS, nA))
    for traj in range(int(num_episodes)):
        # ----------- generate a trajectory ------------ # 
        terminal = 0
        s = np.argmax(np.random.multinomial(1, P_s))
        states = [s]
        actions = []
        rewards = []
        while not terminal:
            a = np.argmax(np.random.multinomial(1, mu[s]))
            actions.append(a)
            r, s, terminal = env.step(s,a)
            rewards.append(r)
            states.append(s)
            # we may stochastically terminate here
            terminal = (terminal or np.random.rand() > P_continue)
        T = len(actions)
        # ----------- END generate a trajectory ------------ # 

        G = 0
        W = 1
        for t in range(T)[::-1]:
            s = states[t]
            a = actions[t]
            r = rewards[t]
            G = r + G * discount
            C[s,a] += W
            Q[s,a] += W / C[s,a] * (G - Q[s,a])
            if greedy:
                pi[s] = np.argmax(Q[s]) # greedification
                if a != pi[s]:
                    break
                W *= 1./mu[s,a]
            else:
                W *= pi[s,a]/mu[s,a]
                if W == 0:
                    break
    
    return C, Q, T


EVMC = off_policy_every_visit_MC


# --------------------------------------------------------------------------
# SECTION 6

# TODO: this uses the old policy representation
def SARSA(mu, env, lr=1., num_episodes=1, Q=None, expected=False):
    if Q is None:
        Q = np.zeros((env.nS, env.nA))
    for episode in range(num_episodes):
        s = sample(env.P_s)
        a = mu.sample(Q[s])
        is_terminal = False
        # TODO: off-policy
        while not is_terminal:
            r, new_s, is_terminal = env.step(s,a)
            if is_terminal:
                target = r
                new_a = a # HACK
            elif expected: # Expected SARSA
                target = r + env.gamma * np.sum(mu.P_a(Q[new_s]) * Q[new_s])
                new_a = mu.sample(Q[new_s])
            else: # SARSA
                new_a = mu.sample(Q[new_s])
                target = r + env.gamma * Q[new_s, new_a]
            Q[s,a] += lr * (target - Q[s,a])
            s, a = new_s, new_a
    return Q


