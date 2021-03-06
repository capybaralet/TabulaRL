import numpy as np
from itertools import product

from environment import TabularMDP

# differences with q-learning environoment:
#   deterministic
#   Gaussian reward function

def R_normal_dist_to_expectation(R):
    return { k : v[0] for k, v in R.iteritems() }

def reward_for_action(state_rewards, action):
    return { (s,action) : reward for s,reward in enumerate(state_rewards) }

def make_sa_rewards(state_rewards, actions=[0]):
    rval = {}
    for action in actions:
        rval.update( { (s,action) : reward for s,reward in enumerate(state_rewards) } )
    return rval

def one_hot(pos, size):
    r = np.zeros(size)
    r[pos] = 1
    return r 


def make_gridworld(grid_width, epLen, rewards, reward_noise=1, multi_chain=False, gotta_move=False):
    """
    gotta_move: if True, then the stay action (a0) leads to (extra) death state with no reward
    """

    assert type(rewards) != list

    nAction=5
    nState = grid_width**2
    def row_and_column(state):
            return state / grid_width, state % grid_width

    if multi_chain: # agent can only move along Ls to the terminal states...
        def transition(state, action):
                row, column = row_and_column(state)
                if action == 1 and row > 0 and column == 0:
                        return state - grid_width
                if action == 2 and column < grid_width - 1 and row > column:
                        return state + 1
                if action == 3 and row < grid_width - 1 and column == 0:
                        return state + grid_width
                if action == 4 and column > 0:
                        return state - 1
                else:
                        return state
    else:
        def transition(state, action):
                row, column = row_and_column(state)
                if action == 1 and row > 0:
                        return state - grid_width
                if action == 2 and column < grid_width - 1:
                        return state + 1
                if action == 3 and row < grid_width - 1:
                        return state + grid_width
                if action == 4 and column > 0:
                        return state - 1
                else:
                        return state


    R_true = {}
    P_true = {}

    for s in xrange(nState):
        for a in xrange(nAction):
            if (s,a) in rewards:
                R_true[s, a] = rewards[s, a]
            else:
                R_true[s, a] = 0

            P_true[s, a] = one_hot(transition(s,a), nState)

    R_true = { k: (v, reward_noise) for k,v in R_true.iteritems() }
    #import ipdb; ipdb.set_trace()
    mdp = make_mdp(nState, nAction, epLen, R_true, P_true, gotta_move=gotta_move)

    mdp.grid_width = grid_width
    mdp.transition = transition
    mdp.row_and_column = row_and_column
    return mdp


def make_longY(nState, epLen, rewards, reward_noise=1):
    """
    An environment where almost all states are unavoidable (and hence not worth querying)
    """
    assert type(rewards) != list
    assert nState > 2

    #nAction=1
    nAction=2
    def transition(state, action):
        #return (state + 1) % nState # long corridor, no branching
        if state == nState - 3: # branch
            if action == 0: # left
                return state + 1
            elif action == 1: # right
                return state + 2
        elif state > nState - 3: # stays put at the ends
            return state 
        else:
            return state + 1

    R_true = {}
    P_true = {}
    for s in xrange(nState):
        for a in xrange(nAction):
            if (s,a) in rewards:
                R_true[s, a] = rewards[s, a]
            else:
                R_true[s, a] = 0
            P_true[s, a] = one_hot(transition(s,a), nState)

    R_true = { k: (v, reward_noise) for k,v in R_true.iteritems() }
    mdp = make_mdp(nState, nAction, epLen, R_true, P_true)
    mdp.transition = transition
    return mdp

def make_random_P(num_states, num_next_states, num_actions=2):
    next_states = [np.random.choice(num_states, num_next_states, replace=False) for ss in range(num_states * num_actions)]
    transition_probs = np.random.uniform(size=(num_states * num_actions, num_next_states))
    transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=1) 
    print transition_probs

    def make_P(transition_probs, next_states):
        rval = np.zeros(num_states)
        for ind, state in enumerate(next_states):
            rval[state] = transition_probs[ind]
        return rval

    P = {}
    for state in range(num_states):
        for action in range(num_actions):
            ind = state * num_actions + action
            P[state, action] = make_P(transition_probs[ind], next_states[ind])

    return P


def make_mdp(nState, nAction, epLen, R, P, reward_noise=None, gotta_move=False):
    assert reward_noise is None
    if gotta_move: # add death state
        env = TabularMDP(nState+1, nAction, epLen)
        # all rewards are 0 in the death state
        R.update({(nState,a): (0,1e-10) for a in range(nAction)})
        # staying put leads to the death state
        P.update({sa: one_hot(np.argmax(P[sa]), nState+1) for sa in P})
        P.update({(s,0): one_hot(nState, nState+1) for s in range(nState)})
        P.update({(nState,a): one_hot(nState, nState+1) for a in range(nAction)})
    else:
        env = TabularMDP(nState, nAction, epLen)
    env.R = R
    env.P = P
    env.reset()
    return env

def bound(x, lower, upper):
    return min(max(x, lower), upper-1)




def make_kchain(chains, epLen, reward_noise=1):
    """"
    Make a world of several chains stuck together.
        LEFT/RIGHT moves agent backwards/forwards along chain
        UP/DOWN moves agent to previous/next chain (if agent is in 0 position along the chain)
        Agent has some chance of being moved backwards along the chain each step. 

    chains : list 
        list of tuples of (length, reward at end)

    make the environment deterministic 
        (and potentially makes the agent know that)
    """
    coords = [ (chain, i) for (chain, (length, end_reward)) in enumerate(chains)
                          for i in range(length)]
    nState = len(coords)
    nChains = len(chains)
    nAction = 5
    stateActions = list(product(range(nState), range(nAction)))

    state_to_coords_map = dict(enumerate(coords))
    coords_to_state_map = { v : k for k,v in state_to_coords_map.iteritems() }

    def coords_to_state(coords): 
        return coords_to_state_map[coords]

    def state_to_coords(state): 
        return state_to_coords_map[state]

    #          stay    next   fore    prev    aft 
    actions = [(0,0), (+1,0), (0,+1), (-1,0), (0,-1)]

    def transition(state, action):
        chain, pos = state_to_coords(state)
        
        dchain, dpos = actions[action]

        if pos == 0: 
            chain = chain + dchain
            chain = bound(chain, 0, len(chains))

        new_pos = pos + dpos

        length, _ = chains[chain]
        noise = 1. / length

        new_pos   = bound(new_pos  , 0, length)
        back_pos  = bound(pos - 1, 0, length)


        new_state   = one_hot(coords_to_state((chain, new_pos)), nState)
        noise_state = one_hot(coords_to_state((chain, back_pos)), nState)
        
        return new_state * (1-noise) + noise_state * noise


    def reward(state, action): 
        chain, pos = state_to_coords(state)
        length, r = chains[chain]

        end = (pos == length - 1)
        noise = (pos == 0) + end

        return end * r, noise
        

    R_true = { (s,a) : (reward(s,a), reward_noise)  for s, a in stateActions }
    P_true = { (s,a) : transition(s,a) for s, a in stateActions }

    mdp = make_mdp(nState, nAction, epLen, R_true, P_true)

    mdp.grid_width = max(length for length, _ in chains)

    def row_and_column(s):
        chain, i = state_to_coords(s)
        return i, chain

    mdp.row_and_column = row_and_column
    mdp.transition = transition
    return mdp
