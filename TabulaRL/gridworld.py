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

def one_hot(pos, size):
    r = np.zeros(size)
    r[pos] = 1
    return r 


def make_gridworld(grid_width, epLen, rewards, reward_noise=1):
    """
    make the environment deterministic 
        (and potentially makes the agent know that)
    """

    assert type(rewards) != list

    nAction=5
    nState = grid_width**2
    def row_and_column(state):
            return state / grid_width, state % grid_width

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

    mdp = make_mdp(nState, nAction, epLen, R_true, P_true, reward_noise)

    mdp.grid_width = grid_width
    mdp.transition = transition
    mdp.row_and_column = row_and_column
    return mdp


def make_mdp(nState, nAction, epLen, R, P, reward_noise, reward_noise_given=False):
    env = TabularMDP(nState, nAction, epLen)
    if not reward_noise_given:
        env.R = { k: (v, reward_noise) for k,v in R.iteritems() }
    else:
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
        

    R_true = { (s,a) : reward(s,a)     for s, a in stateActions }
    P_true = { (s,a) : transition(s,a) for s, a in stateActions }

    mdp = make_mdp(nState, nAction, epLen, R_true, P_true, reward_noise, reward_noise_given=True)

    mdp.grid_width = max(length for length, _ in chains)

    def row_and_column(s):
        chain, i = state_to_coords(s)
        return i, chain

    mdp.row_and_column = row_and_column
    mdp.transition = transition
    eturn mdp
