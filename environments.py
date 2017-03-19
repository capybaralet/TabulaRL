



class ContinuousWindyGridworld(object): # TODO
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
