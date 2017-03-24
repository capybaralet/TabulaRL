"""
WIP RL environments

"""
# TODO1: make 0 always the terminal state
# TODO1: rename S0 --> initial

import numpy
np = numpy

# --------------------------------------------
# ENVIRONMENTS



class FullyConnected(object):
    """ 
    Randomly state transitions, but actions have different rewards 
    All that is needed to get 0 regret is to always choose action 0
    """
    def __init__(self, size=10):
        self.num_states = size + 1
        self.num_actions = size
        self.__dict__.update(locals())
        self.S0 = 0 
        self.terminal = size
        self.rewards = np.dot(np.arange(size).reshape((size, 1)), np.arange(size).reshape((1, size)))
        self.Q = -np.vstack((self.rewards, 
                    np.zeros((1, size))
                 ))

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.terminal:
            return 0, 0, 1
        return np.random.choice(self.num_states), -a * s, 0




class NDGrid(object):
    """ states are represented as integers, I (de)binarize in the step method"""
    def __init__(self, num_dims):
        self.__dict__.update(locals())
        self.num_states = 2 ** self.num_dims + 1
        self.num_actions = self.num_dims
        self.S0 = 0 
        self.terminal = self.num_states  - 1

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
        self.num_states = self.width * self.height + 1
        self.num_actions = 4
        self.S0 = 30
        self.terminal = self.num_states - 1

    def row_and_column(self, state):
            return state / self.width, state % self.width

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if self.tabular:
            if s == self.num_states - 1: # terminal state
                return 0, 0, 1
            if s == 37:
                return self.num_states-1, 0, 0
        else:
            if s == self.num_states - 1: # terminal state
                return self.row_and_column(0), 0, 1
            if s == 37:
                return self.row_and_column(self.num_states-1), 0, 0
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
        self.num_states = grid_width**2 + 1
        self.num_actions = 4
        self.S0 = 0
        self.terminal = self.num_states - 1

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.num_states - 1: # terminal state
            return 0, 0, 1
        if s == self.num_states - 2:
            return self.num_states-1, 0, 0
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


class RandomWalk(object):
    """ 0 is the terminal state """
    def __init__(self, length):
        assert length % 2 == 1
        self.__dict__.update(locals())
        self.num_states = length + 1
        self.num_actions = 2
        self.S0 = self.num_states / 2
        self.terminal = 0

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

# TODO
class ContinuousWindyGridworld(object): # TODO
    """
    Exactly like WindyGridworld, except the states are given in (x,y) coordinates
    """
    def __init__(self):
        self.__dict__.update(locals())
        self.width = 10
        self.height = 7
        self.num_states = self.width * self.height + 1
        self.num_actions = 4
        self.S0 = 30
        self.terminal = self.num_states - 1

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == self.num_states - 1: # terminal state
            return 0, 0, 1
        if s == 37:
            return self.num_states-1, 0, 0
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
