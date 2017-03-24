"""
WIP: want to see if I can use a DNN to do policy evaluation in a single forward pass

The evalnet could take the mdp as an input, or just the policy (design choice)


------------------
N.B.
Since the episode length is finite, the Q-values can be given in closed form as a function of R, P, and pi
So we could (potentially) just do gradient descent on the policy directly!


"""

from src.environment import TabularMDP

#-------------------------------------------------------------------------------
def make_mdp(P, R, epLen):
    """ 
    Just a wrapper for TabularMDP that lets me pass P and R as arrays
    
    P - (s, a, s')
    R - (s, a)
    """
    assert np.all(np.sum(P, axis=1) == 1)
    nState, nAction = P.shape[0], P.shape[1]
    rval = TabularMDP(nState, nAction , epLen)
    # Now initialize R and P
    rval.R = {}
    rval.P = {}
    for state in range(nState):
        for action in range(nAction):
            R[state, action] = (R[state, action], 0)
            P[state, action] = P[state, action]
    return rval


def compute_qVals(R, P, epLen, policy):
    '''
    Compute the Q values for a given R, P estimates

    Args:
        R - R[s,a] = mean rewards
        P - P[s,a] = probability vector of transitions

    Returns:
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep
    '''
    qVals = {}
    qMax = {}

    qMax[epLen] = np.zeros(nState, dtype=np.float32)

    for i in range(epLen):
        j = epLen - i - 1
        qMax[j] = np.zeros(nState, dtype=np.float32)

        for s in range(nState):
            qVals[s, j] = np.zeros(nAction, dtype=np.float32)

            for a in range(nAction):
                qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

            qMax[j][s] = np.max(qVals[s, j])

    return qVals, qMax



