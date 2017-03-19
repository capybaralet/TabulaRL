"""
Implementation based on Sutton Barto 2017, page 162

I will perform 36 experiments total
    ~2 mins / experiment (except range is really 5 exps)
Settings:
    backup_length [1,2,3]
    target_policy x 2
    sigma: range, cap, decay
    sample: 0, 1
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

save_str = 'final_HW1' # TODO


import argparse
parser = argparse.ArgumentParser()
# THESE DON'T CHANGE
parser.add_argument('--debugging', type=int, default=0) # 
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--environment', type=str, default='random_walk')
parser.add_argument('--gamma', type=int, default=1.) #
parser.add_argument('--environment_size', type=int, default=19)
parser.add_argument('--lr', type=float, default=.4) # learning rate
parser.add_argument('--lr_decay', type=float, default=None) #
parser.add_argument('--num_episodes', type=int, default=100) #
parser.add_argument('--num_trials', type=int, default=50) #
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)
# THESE CHANGE
parser = argparse.ArgumentParser()
parser.add_argument('--off_policy', type=str, default='EpsilonGreedy')
parser.add_argument('--backup_length', type=int, default=3)
# TODO: 
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--sigma', type=str, default='average', choices=['0','1','range','decay', 'cap'])
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)


# could use deterministic random seed here
rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

#----------------------------------------
# policy 
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

# --------------------------------------------
# environment
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
# Q-functions (targets) 
# TODO: rename
denominator = 9#float((self.length - 1) / 2)
rw_Q = np.hstack((np.zeros((2,1)), 
                   -np.ones((2,1)), 
                    np.vstack((
                        np.arange(-denominator, denominator - 1) / denominator,
                        (np.arange(-denominator, denominator - 1)  + 2) / denominator)),
                    np.ones((2,1)))).T

w_Q = np.vstack(( np.array([[0,0],[-1,-1],[-1,1]]), np.ones((17,2)) ))

#----------------------------------------
# RUN 

env = RandomWalk(length=environment_size)

# RMS of Q
perfs = np.zeros((len(sigmas), num_trials, num_episodes))

if sigma == 'range':
    sigmas = np.arange(0,1.1,.25)
else: 
    sigmas = [None]

for pp, sig in enumerate(sigmas):
    # set-up sigma
    if sigma == 'cap':
        sigma_fn = lambda x: min(1, 1 / x)
    elif sigma == 'decay':
        sigma_fn = lambda x: episode ** .95
    elif sample:
        sigma_fn = lambda x: rng.rand() < sig
    else:
        sigma_fn = lambda x: sig
        assert sigma_fn(rng.rand()) == sig

    # set-up policies
    if off_policy:
        pi = EpsilonGreedy(0, env)
        ref_Q = w_Q
    else:
        pi = EpsilonGreedy(eps, env)
        ref_Q = rw_Q
    mu = EpsilonGreedy(eps, env)
        
    n = backup_length + 1

    for trial in range(num_trials):

        Q = np.zeros((env.num_states, env.num_actions))

        for episode in range(num_episodes):

            TT = np.inf
            tt = 0
            finished = 0

            s = env.S0
            a = mu.sample(Q[S_t[0]])

            S_t = [s]
            A_t = [a]
            Q_t = [Q[s,a]]
            delta_t = []
            pi_t = [np.inf]
            rho_t = [np.inf]
            sigma_t = [np.inf]
            
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
                        rho_t.append(pi_t[tt+1] /  mu.P_a(Q[s])[a])
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

            perfs[pp,trial,episode] = (np.mean((Q - ref_Q)**2))**.5
            #np.save(save_str + environment + '_' + sigma + '.npy', perfs)

np.save(save_str + environment + '_' + sigma + '___COMPLETE.npy', perfs)


if 1: # plot
    from pylab import *
    figure()
    for qq, sig in zip(perfs, [0,.25,.5,.75,1]):
        plot(np.mean(qq, axis=0), label=sig)
    legend()
    xlabel('episode')
    ylabel('RMS of Q')
    title('error in estimating Q as a function of sigma')
    show()





