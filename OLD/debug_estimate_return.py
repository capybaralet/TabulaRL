"""
BASIC SETUP
------------
We look at n-step TD control, using the Q(sigma) algorithm from Sutton and Barto.

The environment is a deterministic gridworld with -1 reward everywhere.
The agent starts in the top-left corner, and the bottom-right is terminal, so we don't need any reward discounting.

We use epsilon greedy.
The target policy (pi) has a smaller epsilon than the behaviour policy (mu).


TODO
----
1. think about explanations...
    NB: IS < 1 for all except the greedy action!
2. run another experiment with less exploration
3. write it up / think about what to present

How about another algorithm, where we adapt p(sigma) along the way?
And I should probably compare against E[sigma]


NTS
--------
Probably wasn't a great idea to NOT pass rewards in trajectories; it makes it easy to forget them

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




#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backup_length', type=int, default=None)
parser.add_argument('--eps', type=float, default=.1)
parser.add_argument('--grid_width', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-1) # learning rate
parser.add_argument('--lr_decay', type=float, default=1.)#.999)
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--policy', type=str, default='EpsilonGreedy')
args = parser.parse_args()
args_dict = args.__dict__

locals().update(args_dict)

if backup_length is None:
    backup_length = grid_width

# could use deterministic random seed here
rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

#----------------------------------------
# environment

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
# policy 
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


class EpsilonGreedy(object):
    def __init__(self, eps):
        self.eps = eps

    def P_a(self, Q_vals):
        """ probability of taking each action, given their Q-values """
        return self.eps * np.ones(4) / 4. + (1-self.eps) * onehot(np.argmax(Q_vals), 4)

    def sample(self, Q_vals):
        """ sample an action """
        if rng.rand() > self.eps:
            return np.argmax(Q_vals)
        else:
            return rng.choice(len(Q_vals))

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
            s = next_state(s,a)
            if s == grid_width ** 2 - 1:
                finished = 1
                return -step_n
        return -np.inf


#----------------------------------------
# TODO: test
def estimate_return(mu, pi, Q, states, actions, sigma_fn, debug=0):
    """
    A function for computing estimates of the returns with Q-sigma
    """
    assert len(states) == len(actions) == backup_length
    # we go backward through time, i.e. up the backup diagram, aggregating the updates
    for tstep in range(backup_length)[::-1]:
        # update G by one time-step

        s = states[tstep]
        a = actions[tstep]
        if tstep == backup_length - 1:
            G = Q[s,a]

        # compute rho and IS_backup
        Q_vals = Q[s]
        P_a = pi.P_a(Q_vals)
        rho = P_a[a] / mu.P_a(Q_vals)[a]
        IS_backup = rho * G # FIXME: this would fail for backup=1

        # compute tree_backup
        expected_Q = (P_a * Q_vals).sum()
        tree_backup = expected_Q + P_a[a] * (G - Q_vals[a])

        # combine the backups according to sigma
        sigma = sigma_fn(rho)
        reward = -1
        G = reward + sigma * IS_backup + (1 - sigma) * tree_backup 

        if debug:
            print Q
            print s, a, G
            print "IS_backup, tree_backup"
            print IS_backup, tree_backup
            print "G, rho, tstep"
            print G, rho, tstep
            import ipdb; ipdb.set_trace()
    return G

# Get G_{t-1} from G_t
#def backtrack_step(G, mu, pi, Q, s, a, sigma):







#----------------------------------------
# RUN 
orig_lr = lr
all_mu_returns = {}
all_pi_returns = {}
all_Qs = {}
all_Q_diffs = {}

#for prob_tree in np.arange(0, 1.1, .25):
for prob_tree in [0]:
    if False and prob_tree == 0:
        sigma_fn = lambda x: x > 1
    else:
        sigma_fn = lambda x: rng.rand() > prob_tree

    mu_returnz = []
    pi_returnz = []
    Qz = []
    Q_diffz = []

    for trial in range(num_trials):

        print '\n'
        print '\n'
        print prob_tree, trial
        print '\n'
        print '\n'
        lr = orig_lr

        Q = np.zeros((grid_width**2, 4))
        if policy == 'EpsilonGreedy':
            #pi = EpsilonGreedy(eps / 2.)
            pi = EpsilonGreedy(eps=0)
            mu = EpsilonGreedy(eps)
            #mu = Boltzmann(eps / 2.)
        else:
            pi = EpsilonGreedy(eps)
            mu = Boltzmann()
            #mu = Boltzmann(eps / 2.)

        mu_returns = []
        pi_returns = []
        Q_diffs = []

        visit_counts = np.zeros((grid_width**2, 4))

        for episode in range(num_episodes):
            lr *= lr_decay
            states = []
            actions = []
            s = 0
            finished = 0
            step_n = 0
            while not finished: # run an episode
                step_n += 1
                a = mu.sample(Q[s])
                states.append(s)
                actions.append(a)
                visit_counts[s,a] += 1

                # --------- GRID WORLD DYNAMICS ----------
                s = next_state(s,a)

                # "forward view": we update the Q-values of the state-action we visited backup_length time-steps ago, since we now know it's exact update.
                if step_n > backup_length:
                    estimated_return = estimate_return(mu, pi, Q, states[-backup_length:], actions[-backup_length:], sigma_fn=sigma_fn)
                    # apply the updates
                    s_up = states[-backup_length]
                    a_up = actions[-backup_length]
                    Q[s_up, a_up] = (1 - lr) * Q[s_up, a_up] + lr * estimated_return

                if s == grid_width**2 - 1: # the terminal state
                    if 0:
                        for extra_step in range(1):#backup_length): # append extra visits to the terminal state, in order to complete all of the backups
                            step_n += 1
                            states.append(s)
                            actions.append(rng.choice(4))
                            estimated_return = estimate_return(mu, pi, Q, states[-backup_length:], actions[-backup_length:], sigma_fn=sigma_fn)
                            # apply the updates
                            s_up = states[-backup_length]
                            a_up = actions[-backup_length]
                            Q[s_up, a_up] = (1 - lr) * Q[s_up, a_up] + lr * estimated_return
                    finished = 1

            # (off-policy) returns for this episode
            print episode, len(states) - backup_length
            mu_returns.append(-len(states) + backup_length)
            #pi_returns.append(pi.evaluate(Q))
            #Q_diffs.append(np.sum(Q - 
            

        mu_returnz.append(np.array(mu_returns))
        pi_returnz.append(np.array(pi_returns))
        Qz.append(Q)

    all_mu_returns[prob_tree] = mu_returnz
    all_pi_returns[prob_tree] = pi_returnz
    all_Qs[prob_tree] = Qz



#----------------------------------------
# debug
Ss = [0,1,3]
As = [1,2,0]

realQ = []
realQ.append( [3,2,2,3] )
realQ.append( [2,2,1,3] )
realQ.append( [3,1,2,2] )
realQ.append( [1,1,1,1] )
realQ = -A(realQ)


print "TREE"
print "TREE"
print "TREE"
print estimate_return(mu, pi, realQ, [3,3], [0,0], lambda x: 0)
print "IS"
print "IS"
print "IS"
print estimate_return(mu, pi, realQ, [3,3], [0,0], lambda x: 1)
print "TREE"
print "TREE"
print "TREE"
print estimate_return(mu, pi, realQ, Ss[:-1], As[:-1], lambda x: 0)
print "IS"
print "IS"
print "IS"
print estimate_return(mu, pi, realQ, Ss[:-1], As[:-1], lambda x: 1)
print "TREE"
print "TREE"
print "TREE"
print estimate_return(mu, pi, realQ, Ss, As, lambda x: 0)
print "IS"
print "IS"
print "IS"
print estimate_return(mu, pi, realQ, Ss, As, lambda x: 1)

# this is an example optimal policy
pol = [1,2,3,3]

pii = pi

if 0: # plot
    from pylab import *
    figure()
    for kk in all_mu_returns.keys():
        plot(np.mean(all_mu_returns[kk], axis=0), label=kk)
    legend()





