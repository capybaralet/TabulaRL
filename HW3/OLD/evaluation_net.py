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






# NTS: we don't really want to do

# How can we do policy evaluation without a finite horizon?


# this can be used to get a policy from a Q-function, but we also want to do the opposite...
def compute_qVals(R, P, epLen, policy):
    '''
    policy_evaluation
    we represent a policy as 2d-array of (state, P(action | state))
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
                #qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])
                # Instead of taking the qMax, we use the policy's action probabilities
                qVals[s, j][a] = R[s, a] + np.dot(P[s, a], policy[s])

            qMax[j][s] = np.max(qVals[s, j])

    return qVals, qMax











