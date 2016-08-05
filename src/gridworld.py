import numpy as np
from environment import TabularMDP

# differences with q-learning environoment:
#   deterministic
#   Gaussian reward function
def make_gridworld(grid_width, epLen, rewards):
    """
    make the environment deterministic 
        (and potentially makes the agent know that)
    """

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
            R_true[s, a] = (rewards[s], 1)

            P_true[s, a] = np.zeros(nState)
            #deterministic transitions
            P_true[s, a][transition(s, a)] = 1

    env = TabularMDP(nState, nAction, epLen)
    env.R = R_true
    env.P = P_true
    env.reset()

    return env
