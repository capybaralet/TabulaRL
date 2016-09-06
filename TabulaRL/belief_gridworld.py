import numpy as np

"""
TODO: decide to use the existing framework (or not...)

Can we shoe-horn deterministic rewards into it??
    yes!?  just modify update_obs

"""


def compute_qVals(R, P, agent_prior, num_episodes, num_primitive_states, num_primitive_actions, query_cost):
    """
    We have deterministic rewards (and the agent knows this)

    We should compute both how well the agent thinks it will do, and how well it will actually do...

    For now, we assume that the agent knows P

    """
    assert np.all(rewards == 0 or rewards == 1)

    # augment state/action spaces 
    # TODO:
    #   The belief state for each state is either: agent knows its 0, agent knows its 1, or agent thinks its 1 with probability according to its prior
    #   The agent would start off being ignorant (TODO: deal with its prior in the code...)
    num_states = num_primitive_states * 3**num_primitive_states
    num_actions = 2*num_primitive_actions

    def get_primitive_sa(s,a):
        return s / 3**num_primitive_states, a / 2

    def augmented_P(s, a, P):
        ps, pa = get_primitive_sa(s,a)
        new_ps = P[ps, pa]

        if augmented_a % 2 == 1: # query
            query = 1
            # FIXME: updating beliefs must be based on agent's P AND real reward (depending)
            new_s = new_ps * 3**num_primitive_states + (3**ps * (s % 3**ps != 1))
        else: # no query
            query = 0
            new_s = new_ps * 3**num_primitive_states

        return ps, pa, new_s, query

    qVals = {}
    qMax = {}
    qVals_true = {}
    qMax_true = {}

    qMax[num_episodes] = np.zeros(num_states, dtype=np.float32)
    qMax_true[num_episodes] = np.zeros(num_states, dtype=np.float32)
    
    for i in range(num_episodes):
        j = num_episodes - i - 1
        qMax[j] = np.zeros(num_states, dtype=np.float32)
        qMax_true[j] = np.zeros(num_states, dtype=np.float32)

        # TODO: below
        for s in range(num_states):
            qVals[s, j] = np.zeros(num_actions, dtype=np.float32)
            qVals_true[s, j] = np.zeros(num_actions, dtype=np.float32)

            for a in range(num_actions):
                ps, pa, new_s, query = augmented_P(s,a,P)

                qVals_true[s, j][a] = R[ps, pa] + qMax[j + 1][new_s] - query * query_cost

                # TODO: agent believes that if it queries, it will have a 
                qVals[s, j][a] = query * R[get_primitive_sa(s, a)] + np.dot(P_true[s, a], qMax_true[j + 1]) - query * query_cost   

            # agent acts according to what it believes
            a = np.argmax(qVals[s, j])
            # we compute both its estimate of the value of this state/tstep, and the true value
            qMax[j][s] = qVals[s, j][a]
            qMax_true[j][s] = qVals_true[s, j][a]
    
    return qMax_true[0][0], qMax[0][0]












# -------------------------------------
# OLD

def make_gridworld(grid_width, epLen, rewards, reward_noise=1):
    """
    States are belief states
    Actions are (a, query) pairs

    All rewards are deterministically either 0 or 1.
    But the agent plans as if its beliefs are true... 

    The agent should know that the rewards are the same for all (sa)
    We also need to do the reward masking in the agent instead...
    """
    assert type(rewards) != list
    assert type(rewards) != np.array

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


