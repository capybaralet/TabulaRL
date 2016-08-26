import numpy as np

from environment import TabularMDP

# differences with q-learning environoment:
#   deterministic
#   Gaussian reward function

def R_normal_dist_to_expectation(R):
    return { k : v[0] for k, v in R.iteritems() }

def reward_for_action(state_rewards, action):
    return { (s,action) : reward for s,reward in enumerate(state_rewards) }

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

            P_true[s, a] = np.zeros(nState)
            #deterministic transitions
            P_true[s, a][transition(s, a)] = 1

    mdp = make_mdp(nState, nAction, epLen, R_true, P_true, reward_noise)

    mdp.grid_width = grid_width
    mdp.transition = transition
    mdp.row_and_column = row_and_column
    return mdp


def make_mdp(nState, nAction, epLen, R, P, reward_noise):
    env = TabularMDP(nState, nAction, epLen)
    env.R = { k: (v, reward_noise) for k,v in R.iteritems() }
    env.P = P
    env.reset()
    return env

