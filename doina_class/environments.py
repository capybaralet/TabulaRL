"""
WIP RL environments

"""
# TODO1: make 0 always the terminal state
# TODO1: rename S0 --> initial

import numpy
np = numpy

# this repo
from algorithms import get_Q, value_iteration
from utilities import sample



# --------------------------------------------
# ENVIRONMENTS

class MDP(object):
    """
    Unlike Osband's implementation, ours is NOT state-ful (no resetting needed)

    MDP:
        parameters:
            P_s (starting distribution) 
            P, R, gamma
            
            inferred:
                nS, nA, 
                has_terminal (terminal == nS)
        
                optional: V, Q

        methods:
            step()
            get_value_functions()
    """

    def __init__(self, P, R, P_s=None, gamma=1):
        self.__dict__.update(locals())
        self.nS, self.nA = P.shape[0], P.shape[1]
        if self.P_s is None:
            self.P_s = np.zeros(P.shape[0])
            self.P_s[0] = 1
        self.has_terminal = (P.shape[2] == P.shape[0] + 1)

    def step(self, s, a):
        r = self.R[s,a]
        s = sample(self.P[s,a])
        terminal = (s == self.nS)
        return r, s, terminal

    def get_V(self):
        """ set and return self.V using value iteration """
        self.optimal_policy, self.V = value_iteration(self)
        return self.V

    def get_Q(self):
        self.get_V()
        self.Q = get_Q(self, self.V)
        return self.Q
    



# ------------------------------------------
# ------------------------------------------
#def fixed_bandit(size=10, **kwargs):
def lanes1(size, P_terminate=None):
    """
    Returns == first action
    """
    nS = size
    nA = size - 1
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for a in range(nA):
        P[0,a,a+1] = 1
        R[0,a] = a
        for s in range(1, nS):
            P[s,a,s] = 1
            R[s,a] = 0
    P = np.concatenate(( (1 - P_terminate) * P,
                              P_terminate  * np.ones((nS, nA, 1)) ), axis=2)
    return MDP(P,R)

def lanes2(size, P_terminate=None):
    """
    Returns == n * first action
    """
    nS = size
    nA = size - 1
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for a in range(nA):
        P[0,a,a+1] = 1
        R[0,a] = a
        for s in range(1, nS):
            P[s,a,s] = 1
            R[s,a] = s
    P = np.concatenate(( (1 - P_terminate) * P,
                              P_terminate  * np.ones((nS, nA, 1)) ), axis=2)
    return MDP(P,R)

def lanes3(size, P_terminate=None):
    """
    First action teleports the agent to the corresponding state
    Subsequent actions only allow local movement
    """
    nS = size
    nA = size - 1
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for a in range(nA):
        P[0,a,a+1] = 1
        R[0,a] = a
        for s in range(1, nS):
            if a+1 < s and s > 1:
                P[s,a,s-1] = 1
            elif a+1 > s and s < size-1:
                P[s,a,s+1] = 1
            else:
                P[s,a,s] = 1
            R[s,a] = s
    P = np.concatenate(( (1 - P_terminate) * P,
                              P_terminate  * np.ones((nS, nA, 1)) ), axis=2)

    return MDP(P,R)

def lanes4(size, P_terminate=None):
    """
    Every action teleports the agent to the corresponding state.
    Subsequent actions only allow local movement
    """
    nS = size
    nA = size - 1
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    for a in range(nA):
        P[:,a,a+1] = 1
        R[0,a] = a
        for s in range(1, nS):
            R[s,a] = s
    P = np.concatenate(( (1 - P_terminate) * P,
                              P_terminate  * np.ones((nS, nA, 1)) ), axis=2)

    return MDP(P,R)

def random_MDP(size=10, P_terminate=0):
    """ 
    random P and R
    """
    nS = nA = size 
    if P_terminate > 0:
        P = np.concatenate(( (1 - P_terminate) * np.random.dirichlet(1./nS * np.ones(nS), nS * nA).reshape((nS,nA,nS)), 
                            P_terminate  * np.ones((nS, nA, 1)) ), axis=2)
    else:
        P = np.random.dirichlet(1./nS * np.ones(nS), nS * nA).reshape((nS,nA,nS))
    R = np.random.normal(0,1,(nS,nA))

    return MDP(P,R)


def many_actions(size, P_terminate=0):



def fully_connected(size=10):
    """ 
    State transitions are uniform random, but actions have different rewards 
    All that is needed to get 0 regret is to always choose action 0
    """
    # has_terminal == 1
    P = np.ones((size, size, size+1)) * 1./(size+1)
    R = np.array([-a * s for s in range(size) for a in range(size)]).reshape((size, size))

    return MDP(P,R)

def grid_world(size, has_terminal=True):
    assert has_terminal
    # has_terminal == 1
    P = np.zeros((size**2, 4, size**2 + 1))
    P[-1,:,-1] = 1
    for s in range(size**2 - 1): # don't update bottom-right, it leads to terminal state!
        # a=0  --  up
        if s / size == 0:
            P[s, 0, s] = 1
        else:
            P[s, 0, s - size] = 1
        # a=1  --  right
        if s % size == size - 1:
            P[s, 1, s] = 1
        else:
            P[s, 1, s + 1] = 1
        # a=2  --  down
        if s / size == size - 1:
            P[s, 2, s] = 1
        else:
            P[s, 2, s + size] = 1
        # a=3  --  left
        if s % size == 0:
            P[s, 3, s] = 1
        else:
            P[s, 3, s - 1] = 1

    R = -1 * np.ones((size**2, 4))
    R[-1] = 0

    return MDP(P,R)
    
def random_walk(size=19):
    """
    The environment from the Alberta paper
    """

    assert size % 2 == 1
    P_s = np.zeros(size)
    P_s[size / 2] = 1

    # has_terminal == 1
    P = np.zeros((size, 2, size+1))
    for n in range(size):
        P[n][0][n-1] = 1
        P[n][1][n+1] = 1
    # for terminal state:
    P[0,:] = 0
    P[0,:,-1] = 1
    P[-1,:] = 0
    P[-1,:,-1] = 1

    R = np.zeros((size, 2)) # no reward for terminal
    R[0] = -1
    R[-1] = 1

    return MDP(P, R, P_s=P_s)



env_dict = {'lanes1':lanes1,'lanes2':lanes2,'lanes3':lanes3,'lanes4':lanes4,
            'fully_connected':fully_connected,
            'grid_world':grid_world,
            'random_MDP':random_MDP,
            'random_walk':random_walk}



# ------------------------------------------
# ------------------------------------------
# --------------------------------------------
# deprecated (TODO: rm)



# FIXME: -1 is terminal according to P (as it should be!)
class RandomWalk(object):
    """ 0 is the terminal state """
    def __init__(self, length):
        assert length % 2 == 1
        self.__dict__.update(locals())
        self.nS = length + 1
        self.nA = 2
        self.S0 = self.nS / 2
        self.terminal = 0
        
        self.R = np.zeros((length, self.nA)) # no reward for terminal
        self.R[0] = -1
        self.R[-1] = 1

        self.P = np.zeros((self.length, 2, self.length+1))
        for n in range(self.length):
            self.P[n][0][n-1] = 1
            self.P[n][1][n+1] = 1
        # for terminal state...
        self.P[0,:] = 0
        self.P[0,:,-1] = 1
        self.P[-1,:] = 0
        self.P[-1,:,-1] = 1
        #
        self.gamma = 1

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == 0: # terminal state
            return 0, 0, 1
        if s == 1:
            return 0, -1, 0
        if s == self.length:
            return 0, 1, 0
        if a == 0: # left
            return s-1, 0, 0
        if a == 1: # right
            return s+1, 0, 0



class FullyConnected(object):
    """ 
    Randomly state transitions, but actions have different rewards 
    All that is needed to get 0 regret is to always choose action 0
    """
    def __init__(self, size=10):
        self.nS = size + 1
        self.nA = size
        self.__dict__.update(locals())
        self.S0 = 0 
        self.terminal = size
        self.rewards = np.dot(np.arange(size).reshape((size, 1)), np.arange(size).reshape((1, size)))
        self.Q = -np.vstack((self.rewards, 
                    np.zeros((1, size))
                 ))
        
        # s, a, s'
        self.P = np.ones((size, size, size+1)) * 1./(size+1)
        self.R = np.array([-a * s for s in range(size) for a in range(size)]).reshape((size, size))

    """
    def P(self, s, a):
        return np.random.choice(self.nS)

    def R(self, s, a):
        return -a * s
    """

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.terminal:
            return 0, 0, 1
        return np.random.choice(self.nS), -a * s, 0




class myMDP(object):

    def __init__(self, P, R, gamma=None):
        self.__dict__.update(locals())
        self.nS, self.nA = P.shape[0], P.shape[1]
        self.S0 = 0
        self.terminal = None
        if gamma is None:
            self.gamma = 1

    def step(self, s, a):
        new_s = sample(self.P[s,a])
        r = self.R[s,a]
        terminal = sample([self.gamma, 1-self.gamma])
        #import ipdb; ipdb.set_trace()
        return new_s, r, terminal


# --------------------------------------------
# TODO: redo as MDP


class NDGrid(object):
    """ states are represented as integers, I (de)binarize in the step method"""
    def __init__(self, num_dims):
        self.__dict__.update(locals())
        self.nS = 2 ** self.num_dims + 1
        self.nA = self.num_dims
        self.S0 = 0 
        self.terminal = self.nS  - 1

    def binarize(self, state):
            return np.array([(state / nn) % 2 for nn in 2**np.arange(0, self.num_dims)])

    def unbinarize(self, state):
            return (state * 2**np.arange(0, self.num_dims)).sum()

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.terminal:
            return 0, 0, 1
        elif s == self.terminal - 1:
            return self.terminal, 0, 0
        new_s = self.binarize(s)
        new_s[a] = (new_s[a] + 1) % 2
        new_s = self.unbinarize(new_s)
        return new_s, -1, 0


class WindyGridWorld(object):
    def __init__(self, tabular=True):
        self.__dict__.update(locals())
        self.width = 10
        self.height = 7
        self.nS = self.width * self.height + 1
        self.nA = 4
        self.S0 = 30
        self.terminal = self.nS - 1

    def row_and_column(self, state):
            return state / self.width, state % self.width

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if self.tabular:
            if s == self.nS - 1: # terminal state
                return 0, 0, 1
            if s == 37:
                return self.nS-1, 0, 0
        else:
            if s == self.nS - 1: # terminal state
                return self.row_and_column(0), 0, 1
            if s == 37:
                return self.row_and_column(self.nS-1), 0, 0
        if a == 0: # up
            if s / self.width == 0:
                new_s = s
            else:
                new_s = s - self.width
        if a == 1: # right
            if s % self.width == self.width-1:
                new_s = s
            else:
                new_s = s + 1
        if a == 2: # down
            if s / self.width == self.height-1:
                new_s = s
            else:
                new_s = s + self.width
        if a == 3: # left
            if s % self.width == 0:
                new_s = s
            else:
                new_s = s - 1

        # add wind
        if new_s % 10 in [3,4,5,6,7,8] and new_s > 10:
            new_s -= 10
        if new_s % 10 in [6,7] and new_s > 10:
            new_s -= 10

        if self.tabular:
            return new_s, -1, 0
        else:
            return self.row_and_column(new_s), -1, 0
        

class GridWorld(object):
    def __init__(self, grid_width):
        self.__dict__.update(locals())
        self.nS = grid_width**2 + 1
        self.nA = 4
        self.S0 = 0
        self.terminal = self.nS - 1

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.nS - 1: # terminal state
            return 0, 0, 1
        if s == self.nS - 2:
            return self.nS-1, 0, 0
        if a == 0: # up
            if s / self.grid_width == 0:
                return s, -1, 0
            else:
                return s - self.grid_width, -1, 0
        if a == 1: # right
            if s % self.grid_width == self.grid_width-1:
                return s, -1, 0
            else:
                return s + 1, -1, 0
        if a == 2: # down
            if s / self.grid_width == self.grid_width-1:
                return s, -1, 0
            else:
                return s + self.grid_width, -1, 0
        if a == 3: # left
            if s % self.grid_width == 0:
                return s, -1, 0
            else:
                return s - 1, -1, 0

# TODO
class ContinuousWindyGridworld(object): # TODO
    """
    Exactly like WindyGridworld, except the states are given in (x,y) coordinates
    """
    def __init__(self):
        self.__dict__.update(locals())
        self.width = 10
        self.height = 7
        self.nS = self.width * self.height + 1
        self.nA = 4
        self.S0 = 30
        self.terminal = self.nS - 1

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.nS - 1: # terminal state
            return 0, 0, 1
        if s == 37:
            return self.nS-1, 0, 0
        if a == 0: # up
            if s / self.width == 0:
                new_s = s
            else:
                new_s = s - self.width
        if a == 1: # right
            if s % self.width == self.width-1:
                new_s = s
            else:
                new_s = s + 1
        if a == 2: # down
            if s / self.width == self.height-1:
                new_s = s
            else:
                new_s = s + self.width
        if a == 3: # left
            if s % self.width == 0:
                new_s = s
            else:
                new_s = s - 1

        # add wind
        if new_s % 10 in [3,4,5,6,7,8] and new_s > 10:
            new_s -= 10
        if new_s % 10 in [6,7] and new_s > 10:
            new_s -= 10

        return new_s, -1, 0
