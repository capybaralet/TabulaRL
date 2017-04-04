"""
WIP based on Sutton Barto

It doesn't work (or does it???)

"""

#TODO: backup_length=1 doesn't work
#FIXME: I think I should have 

import numpy
np = numpy
import numpy.random

def onehot(x, length):
    rval = np.zeros(length)
    rval[x] = 1
    return rval

def softmax(w):
    w = numpy.array(w)
    if len(w.shape) == 1:
        maxes = np.max(w)
        e = numpy.exp(w - maxes)
        dist = e / numpy.sum(e)
        return dist
    maxes = numpy.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = numpy.exp(w - maxes)
    dist = e / numpy.sum(e, axis=1, keepdims=True)
    return dist

def print_fn(thing):
    print thing

#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backup_length', type=int, default=4)
parser.add_argument('--debugging', type=int, default=1)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='random_walk')
parser.add_argument('--gamma', type=int, default=1.)
parser.add_argument('--environment_size', type=int, default=19)
parser.add_argument('--lr', type=float, default=.4) # learning rate
parser.add_argument('--lr_decay', type=float, default=None)
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--num_trials', type=int, default=1)
#parser.add_argument('--policy', type=str, default='EpsilonGreedy')
parser.add_argument('--policy', type=str, default='MoveLeft')
args = parser.parse_args()
args_dict = args.__dict__

locals().update(args_dict)
orig_lr = lr

if backup_length is None:
    backup_length = grid_width

# could use deterministic random seed here
rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

#----------------------------------------
# policy 
# TODO: nA
class Boltzmann(object):
    def __init__(self, eps, inv_temp=.1):
        self.eps = eps
        self.inv_temp = inv_temp

    def P_a(self, Q_vals):
        """ probability of taking each action, given their Q-values """
        B_probs = softmax(self.inv_temp * Q_vals)
        # mixture of Boltzmann + epsilon greedy
        return self.eps * np.ones(4) / 4. + (1-self.eps) * B_probs

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
        return self.eps * np.ones(self.env.nA) / float(self.env.nA) + (1-self.eps) * onehot(np.argmax(Q_vals), self.env.nA)

    def sample(self, Q_vals):
        """ sample an action """
        if rng.rand() > self.eps:
            return np.argmax(Q_vals)
        else:
            return rng.choice(len(Q_vals))

    # TODO: rm
    def evaluate(self, Q, max_steps=np.inf):
        """ 
        Run an entire episode and compute the returns
        """
        s = 0
        finished = 0
        step_n = 0
        while step_n < max_steps: # run an episode
            step_n += 1
            a = self.sample(Q[s])

            # --------- GRID WORLD DYNAMICS ----------
            if a == 0: # up
                if s / grid_width == 0:
                    new_s = s
                else:
                    new_s = s - grid_width
            if a == 1: # right
                if s % grid_width == grid_width-1:
                    new_s = s
                else:
                    new_s = s + 1
            if a == 2: # down
                if s / grid_width == grid_width-1:
                    new_s = s
                else:
                    new_s = s + grid_width
            if a == 3: # left
                if s % grid_width == 0:
                    new_s = s
                else:
                    new_s = s - 1
            s = new_s

            if s == grid_width ** 2 - 1:
                finished = 1
                return -step_n
        return -np.inf

# --------------------------------------------
# ENVIRONMENTS
class GridWorld(object):
    def __init__(self, grid_width):
        self.__dict__.update(locals())
        self.nS = grid_width**2
        self.nA = 4
        self.realQ = 0 # TODO
        self.S0 = 0

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        if s == 0:
            return 0, 0, 1
        if s == self.nS:
            return 0, 0, 0
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
        self.nS = length + 1
        self.nA = 2
        self.S0 = self.nS / 2

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


# --------- GRID WORLD DYNAMICS ----------
def next_state(s,a):
    if a == 0: # up
        if s / grid_width == 0:
            return s
        else:
            return s - grid_width
    if a == 1: # right
        if s % grid_width == grid_width-1:
            return s
        else:
            return s + 1
    if a == 2: # down
        if s / grid_width == grid_width-1:
            return s
        else:
            return s + grid_width
    if a == 3: # left
        if s % grid_width == 0:
            return s
        else:
            return s - 1

#----------------------------------------
def estimate_return(mu, pi, Q, states, actions, sigma_fn):
    """
    A function for computing estimates of the returns with Q-sigma
    """
    estimated_returns = 0
    for tstep in range(backup_length)[::-1]:
        s = states[tstep]
        a = actions[tstep]
        Q_vals = Q[s]
        IS_weight = pi.P_a(Q_vals)[a] / mu.P_a(Q_vals)[a]
        if sigma_fn(IS_weight): # use IS
            estimated_returns += Q_vals[a]
            estimated_returns *= IS_weight
        else: # use tree backup
            estimated_returns += (Q_vals * pi.P_a(Q_vals)).sum()
    if s != grid_width**2 - 1:
        estimated_returns -= 1
    return estimated_returns


#----------------------------------------
# RUN 
orig_lr = lr
all_mu_returns = {}
all_pi_returns = {}
all_Qs = {}
all_Q_diffs = {}

if environment == 'grid_world':
    env = GridWorld(grid_width=environment_size)
elif environment == 'random_walk':
    env = RandomWalk(length=environment_size)

#for prob_tree in np.arange(0, 1.1, .25):
for prob_tree in [1]:
    if False and prob_tree == 0:
        sigma_fn = lambda x: x > 1
    else:
        sigma_fn = lambda x: rng.rand() > prob_tree

    mu_returnz = []
    pi_returnz = []
    Qz = []
    Q_diffz = []

    for trial in range(num_trials):

        print '\n\n', prob_tree, trial, '\n\n'

        if policy == 'EpsilonGreedy':
            pi = EpsilonGreedy(eps / 2., env)
            pi = EpsilonGreedy(eps, env)
            mu = EpsilonGreedy(eps, env)
        elif policy == 'MoveLeft':
            pi = MoveLeft()
            mu = MoveLeft()
        else: # TODO
            pi = EpsilonGreedy(eps)
            mu = Boltzmann()
            #mu = Boltzmann(eps / 2.)

        # TODO: logging
        mu_returns = []
        pi_returns = []
        Q_diffs = []

        # ----------- BEGIN ------------ #
        Q = np.zeros((env.nS, env.nA))
        n = backup_length


        for episode in range(num_episodes):
            S_t = []
            A_t = []
            Q_t = []
            delta_t = []
            pi_t = [np.inf]
            rho_t = [np.inf]
            sigma_t = [np.inf]

            #mu.eps = eps / (.01*episode + 1)
            #lr = orig_lr / (episode * lr_decay + orig_lr)
            #print "(Q**2).sum()", (Q**2).sum()
            s = env.S0
            S_t.append(s)
            a = mu.sample(Q[S_t[0]])
            A_t.append(a)
            Q_t.append(Q[s,a])
            TT = np.inf
            tt = 0
            finished = 0
            
            # run Q(sigma)
            while not finished:
                if tt < TT: # if not terminal, get next state and action
                    s, r, is_terminal = env.step(s,a)
                    # tt + 1
                    S_t.append(s)
                    if is_terminal:
                        TT = tt + 1
                        # tt
                        delta_t.append(r - Q_t[tt])
                    else:
                        a = mu.sample(Q[s])
                        # tt + 1
                        A_t.append(a)
                        Q_t.append(Q[s,a])
                        # MOVED
                        pi_t.append(pi.P_a(Q[s])[a])
                        rho_t.append(pi_t[tt+1] /  mu.P_a(Q[s])[a])
                        sigma_t.append(sigma_fn(rho_t))
                        # END MOVED
                        # tt
                        delta_t.append(r - Q_t[tt] + gamma * sigma_t[tt+1]*Q_t[tt+1] + gamma * (1-sigma_t[tt+1])*np.sum(pi.P_a(Q[s]) * Q[s]))
                        if debugging:
                            print "tt", tt
                            print "delta_t", delta_t
                tau = tt - n + 1
                # FIXME: updating the wrong (s,a)!!!
                # ^ this is some sort of off-by-one error... 
                if tau >= 0: # update Q[S_tau, A_tau]
                    print "\nupdating..."
                    print "tau", tau
                    if 0:#tau == 2:
                        import ipdb; ipdb.set_trace()
                    rho = 1
                    E = 1
                    G = Q_t[tau]
                    for k in range(tau, min(tt, TT - 1)):
                        if debugging:
                            print "G", G
                        #print "tau, k=", tau, k
                        G += E * delta_t[k]
                        E *= gamma * ((1 - sigma_t[k+1]) * pi_t[k+1] + sigma_t[k+1])
                        #rho *= (1 - sigma_t[km] + sigma_t[km] * rho_t[km])
                        rho *= (1 - sigma_t[k+1] + sigma_t[k+1] * rho_t[k+1])
                        if 0: #print
                            print "k, E, G, rho"
                            print k, E, G, rho
                    #S_tau, A_tau = S_t[taum1], A_t[taum1]
                    S_tau, A_tau = S_t[tau], A_t[tau]
                    if debugging and episode > 0:
                        import ipdb; ipdb.set_trace()
                    if 1: #debugging:
                        print "S_tau, A_tau", S_tau, A_tau
                        print "G, Q[S_tau, A_tau]", G, Q[S_tau, A_tau]
                        #import ipdb; ipdb.set_trace()
                    if not S_tau == 0: # don't update Q for terminal state
                        Q[S_tau, A_tau] += lr * rho * (G - Q[S_tau, A_tau])
                    if np.isnan(np.sum(Q)):
                        import ipdb; ipdb.set_trace()
                    print "done updating! \n"

                tt += 1

                if 0:#debugging:
                    print "s,tt,tau,TT", S_t[tt-1],tt-1,tau,TT

                if tau == TT - 1: # FIXME: taking one too many steps?
                    if debugging:
                        print allS_t
                        print allA_t
                        print Q
                    print "FINISHED AT", tt, '\n\n'
                    finished = 1


        # ---------- END ---------- #

        # TODO: below

    """
            # (off-policy) returns for this episode
            print episode, len(states) - backup_length
            mu_returns.append(-len(states) + backup_length)
            pi_returns.append(pi.evaluate(Q))
            #Q_diffs.append(np.sum(Q - 
            

        mu_returnz.append(np.array(mu_returns))
        pi_returnz.append(np.array(pi_returns))
        Qz.append(Q)

    all_mu_returns[prob_tree] = mu_returnz
    all_pi_returns[prob_tree] = pi_returnz
    all_Qs[prob_tree] = Qz
    """



if 0: # plot
    from pylab import *
    figure()
    for kk in all_mu_returns.keys():
        plot(np.mean(all_mu_returns[kk], axis=0), label=kk)
    legend()





