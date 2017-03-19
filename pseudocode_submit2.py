"""
Implementation based on Sutton Barto 2017, page 162
"""


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

exps = {}
exps['rw'] = {}
exps['w'] = {}
exps['grid'] = {}

save_str = 'final_HW1' # TODO

"""
24 experiments total
    ~2 mins / experiment (except range is really 5 exps)
Settings:
    backup_length [2,3]
    target_policy x 2
    sigma: (sample, average) x (range, cap, decay)
"""

import argparse
parser = argparse.ArgumentParser()
# DON'T CHANGE THESE!
parser.add_argument('--debugging', type=int, default=0) # 
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='random_walk')
parser.add_argument('--gamma', type=int, default=1.) #
parser.add_argument('--environment_size', type=int, default=19)
parser.add_argument('--lr', type=float, default=.4) # learning rate
parser.add_argument('--lr_decay', type=float, default=None) #
parser.add_argument('--num_episodes', type=int, default=100) #
parser.add_argument('--num_trials', type=int, default=50) #
parser.add_argument('--policy', type=str, default='EpsilonGreedy')
parser.add_argument('--target_policy', type=str, default='Greedy')
#
parser.add_argument('--backup_length', type=int, default=3) # FIXME: offby1
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--sigma', type=str, default='range', choices=['0','1','range','decay', 'cap', 'algorithm1',  'algorithm2', 'retrace']) # TODO: I'm not sure this is actually retrace
# PHASES (TODO)
parser.add_argument('--run_setup', type=int, default=1)
parser.add_argument('--run_training', type=int, default=1)
parser.add_argument('--run_plotting', type=int, default=1)

#parser.add_argument('--policy', type=str, default='MoveLeft')
args = parser.parse_args()
args_dict = args.__dict__

locals().update(args_dict)
orig_lr = lr

#assert environment == 'random_walk'


if backup_length is None:
    backup_length = grid_width

# could use deterministic random seed here
rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

#----------------------------------------
# Evaluation (TODO: problem is that the target-policy may never reach a terminal state...)
def evaluate(policy, environment):
    returns = 0  

    def step(self, s, a):
        """ returns s_{t+1}, r_t, and is_terminal_{t+1} """
        pass

#----------------------------------------
# policy 
# TODO: num_actions
class Boltzmann(object):
    def __init__(self, eps, env, inv_temp=1.):
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

class WindyGridWorld(object):
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


#----------------------------------------
# ground-truth Q-functions (targets) 
denominator = float((environment_size - 1) / 2)
rw_Q = np.hstack((np.zeros((2,1)), 
                   -np.ones((2,1)), 
                    np.vstack((
                        np.arange(-denominator, denominator - 1) / denominator,
                        (np.arange(-denominator, denominator - 1)  + 2) / denominator)),
                    np.ones((2,1)))).T

w_Q = np.vstack(( np.array([[0,0],[-1,-1],[-1,1]]), np.ones((environment_size-2,2)) ))

w_Q_1 = np.array([[ 0.        ,  0.        ],
       [-1.        , -1.        ],
       [-1.        ,  0.99445983],
       [ 0.89473684,  0.99970841],
       [ 0.99445983,  0.99998465],
       [ 0.99970841,  0.99999919],
       [ 0.99998465,  0.99999996],
       [ 0.99999919,  1.        ],
       [ 0.99999996,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ],
       [ 1.        ,  1.        ]])

w_Q_5 = np.array([[ 0.        ,  0.        ],
       [-1.        , -1.        ],
       [-1.        ,  0.77777778],
       [ 0.33333334,  0.92592593],
       [ 0.77777778,  0.97530865],
       [ 0.92592593,  0.99176955],
       [ 0.97530865,  0.99725652],
       [ 0.99176955,  0.99908551],
       [ 0.99725652,  0.99969517],
       [ 0.99908551,  0.99989839],
       [ 0.99969517,  0.99996613],
       [ 0.99989839,  0.99998872],
       [ 0.99996613,  0.99999624],
       [ 0.99998872,  0.99999875],
       [ 0.99999624,  0.99999959],
       [ 0.99999875,  0.99999987],
       [ 0.99999959,  0.99999996],
       [ 0.99999987,  0.99999999],
       [ 0.99999996,  1.        ],
       [ 1.        ,  1.        ]]) 


#----------------------------------------
# RUN 
orig_lr = lr
all_mu_returns = {}
all_pi_returns = {}
all_Qs = {}
all_Q_diffs = {}


if environment == 'grid_world':
    env = GridWorld(grid_width=environment_size)
elif environment == 'windy_grid_world':
    env = WindyGridWorld()
elif environment == 'random_walk':
    env = RandomWalk(length=environment_size)


if sigma == '0':
    sigmas = [0]
elif sigma == '1' or 'retrace':
    sigmas = [1]
elif sigma == 'average':
    sigmas = np.arange(0,1.1,.25)
elif sigma == 'sample':
    sigmas = np.arange(0,1.1,.25)
else:
    sigmas = [None]


if sigma == 'sample':
    sample = 1
else:
    sample = 0

perfs = np.zeros((len(sigmas), num_trials, num_episodes))



for pp, sig in enumerate(sigmas):
    if sigma == 'cap':
        sigma_fn = lambda x: min(1, 1 / x)
    elif sigma == 'algorithm1':
        sigma_fn = lambda x: 1 if x <= 1 else 0
    elif sigma == 'algorithm2':
        sigma_fn = lambda x,y: 0 if y == 0 else min(1, x / (1-x) * (1-y) / y)
    elif sigma == 'decay':
        sigma_fn = lambda x: episode ** .1
    elif sample:
        sigma_fn = lambda x: rng.rand() < sig
    else:
        sigma_fn = lambda x: sig
        assert sigma_fn(rng.rand()) == sig

    for trial in range(num_trials):
        if policy == 'EpsilonGreedy':
            if target_policy == 'EpsilonGreedy':#environment == 'random_walk':
                pi = EpsilonGreedy(eps, env)
                ref_Q = rw_Q
            else:
                g_eps = .1
                pi = EpsilonGreedy(g_eps, env)
                if g_eps == .1:
                    ref_Q = w_Q_1
                if g_eps == .5:
                    ref_Q = w_Q_5
                else:
                    ref_Q = w_Q
            if environment == 'grid_world':
                ref_Q = 0
            mu = EpsilonGreedy(eps, env)
        elif policy == 'MoveLeft':
            pi = MoveLeft()
            mu = MoveLeft()
        else: # TODO
            #pi = EpsilonGreedy(0, env)
            pi = Boltzmann(0, env)
            mu = Boltzmann(0, env)
            #mu = Boltzmann(eps / 2.)

        # ----------- BEGIN ------------ #
        Q = np.zeros((env.num_states, env.num_actions))
        n = backup_length

        all_sigma_t = []

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
                print tt
                if tt < TT: # if not terminal, get next state and action
                    s, r, is_terminal = env.step(s,a)
                    S_t.append(s)
                    if is_terminal:
                        TT = tt + 1
                        delta_t.append(r - Q_t[tt])
                    else:
                        a = mu.sample(Q[s])
                        A_t.append(a)
                        Q_t.append(Q[s,a])
                        pi_t.append(pi.P_a(Q[s])[a])
                        if sigma == 'retrace':
                            rho_t.append(min(pi_t[tt+1] /  mu.P_a(Q[s])[a], 1))
                        else:
                            rho_t.append(pi_t[tt+1] /  mu.P_a(Q[s])[a])
                        if sigma == 'algorithm2':
                            if debugging:
                                muu = mu.P_a(Q[s])[a]
                                pii = pi.P_a(Q[s])[a]
                                print muu, pii
                            sigma_t.append(sigma_fn(mu.P_a(Q[s])[a], pi_t[tt+1]))
                        else:
                            sigma_t.append(sigma_fn(rho_t[-1]))
                        delta_t.append(r - Q_t[tt] + gamma * sigma_t[tt+1]*Q_t[tt+1] + gamma * (1-sigma_t[tt+1])*np.sum(pi.P_a(Q[s]) * Q[s]))
                        if debugging:
                            print "tt", tt
                            print "delta_t", delta_t
                tau = tt - n + 1
                if tau >= 0: # update Q[S_tau, A_tau]
                    rho = 1
                    E = 1
                    G = Q_t[tau]
                    for k in range(tau, min(tt, TT - 1)):
                        if debugging:
                            print "k, tau", k, tau
                            print "G", G
                        G += E * delta_t[k]
                        E *= gamma * ((1 - sigma_t[k+1]) * pi_t[k+1] + sigma_t[k+1])
                        rho *= (1 - sigma_t[k+1] + sigma_t[k+1] * rho_t[k+1])
                    S_tau, A_tau = S_t[tau], A_t[tau]
                    if not S_tau == env.terminal: # don't update Q for terminal state
                        Q[S_tau, A_tau] += lr * rho * (G - Q[S_tau, A_tau])

                tt += 1

                if tau == TT - 1:
                    print "FINISHED AT", tt
                    finished = 1

            all_sigma_t += sigma_t
            if 1:# environment == 'random_walk':
                perfs[pp,trial,episode] = (np.mean((Q - ref_Q)**2))**.5
            else:
                pass
                #perfs[pp,trial,episode] = returns

            np.save('HW1b_perfs_' + environment + '_' + sigma + '.npy', perfs)

        # ---------- END ---------- #

np.save('HW1b_perfs_' + environment + '_' + sigma + '___COMPLETE', perfs)
#np.save('HW1_Q_errs_' + sigma + '.npy', Q_errs)


if 1: # plot
    from pylab import *
    figure()
    for qq, sig in zip(perfs, [0,.25,.5,.75,1]):
        plot(np.mean(qq, axis=0), label=sig)
    legend()
    show()
    xlabel('episode')
    ylabel('RMS of Q')
    title('error in estimating Q as a function of sigma')





