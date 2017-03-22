# TODO: random seed

import numpy
np = numpy

class Boltzmann(object):
    def __init__(self, eps, env, inv_temp=1., rng=None):
        self.eps = eps
        self.env = env
        self.inv_temp = inv_temp

    def P_a(self, Q_vals):
        """ probability of taking each action, given their Q-values """
        B_probs = softmax(self.inv_temp * Q_vals)
        # mixture of Boltzmann + epsilon greedy
        return self.eps * np.ones(self.env.num_actions) / float(self.env.num_actions) + (1-self.eps) * B_probs

    def sample(self, Q_vals):
        """ sample an action """
        return np.argmax(rng.multinomial(1, self.P_a(Q_vals)))


class MoveLeft(object):
    def P_a(self, Q_vals):
        """ probability of taking each action, given their Q-values """
        return [1,0]

    def sample(self, Q_vals):
        """ sample an action """
        return 0


class EpsilonGreedy(object):
    def __init__(self, eps, env):
        self.eps = eps
        self.env = env

    def P_a(self, Q_vals):
        """ probability of taking each action, given their Q-values """
        return self.eps * np.ones(self.env.num_actions) / float(self.env.num_actions) + (1-self.eps) * onehot(np.argmax(Q_vals), self.env.num_actions)

    def sample(self, Q_vals):
        """ sample an action """
        if rng.rand() > self.eps:
            return np.argmax(Q_vals)
        else:
            return rng.choice(len(Q_vals))
