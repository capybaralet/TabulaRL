"""
BASIC SETUP
------------
We look at n-step TD control, using the Q(sigma) algorithm from Sutton and Barto.

The environment is a deterministic gridworld with -1 reward everywhere.
The agent starts in the top-left corner, and the bottom-right is terminal, so we don't need any reward discounting.

We use epsilon greedy.
The target policy (pi) has a smaller epsilon than the behaviour policy (mu).
"""

import numpy
np = numpy
import numpy.random

def onehot(x, length):
    rval = np.zeros(length)
    rval[x] = 1
    return rval

#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backup_length', type=int, default=None)
parser.add_argument('--eps', type=float, default=.1)
parser.add_argument('--grid_width', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3) # learning rate
parser.add_argument('--lr_decay', type=float, default=.9999) # learning rate
parser.add_argument('--num_episodes', type=int, default=10000)
parser.add_argument('--sigma_fn', type=str, default='tree', choices=['IS', 'random', 'thresh', 'tree'])
#
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
args_dict = args.__dict__

# SET-UP SAVING
if args_dict['save']:
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    flags = [ff for ff in flags if not ff.startswith('save_dir')]
    save_dir = args_dict.pop('save_dir')
    save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
    args_dict['save_path'] = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')
    print(save_path)
    #assert False

locals().update(args_dict)

if backup_length is None:
    backup_length = grid_width

# RANDOM SEED
if seed is not None:
    np.random.seed(seed)  # for reproducibility
    rng = numpy.random.RandomState(seed)
else:
    rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

if sigma_fn == 'IS':
    sigma_fn = lambda x: 1
if sigma_fn == 'random':
    sigma_fn = lambda x: rng.rand()
if sigma_fn == 'thresh':
    sigma_fn = lambda x: x < 1
if sigma_fn == 'tree':
    sigma_fn = lambda x: 0

#----------------------------------------
# visualization

import matplotlib.pyplot as plt
def plotQ(Q):
    action_vals = (Q.T).reshape((4, grid_width, -1))
    plt.figure()
    for n in range(4):
        plt.subplot(2,2,n+1)
        plt.imshow(action_vals[n], cmap='Greys', interpolation='none')


#----------------------------------------
# policy 
class EpsilonGreedy(object):
    def __init__(self, eps):
        self.eps = eps

    def P_a(self, Q_vals):
        return self.eps * np.ones(4) / 4. + (1-self.eps) * onehot(np.argmax(Q_vals), 4)

    def sample(self, Q_vals):
        if rng.rand() > self.eps:
            return np.argmax(Q_vals)
        else:
            return rng.choice(len(Q_vals))


#----------------------------------------
# ...


def estimate_return(mu, pi, Q, states, actions, sigma_fn=sigma_fn):
    #assert len(states) == len(actions) == len(sigmas) == backup_length
    assert len(states) == len(actions) == backup_length
    estimated_returns = 0
    for tstep in range(backup_length)[::-1]:
        #print "sigmas", sigmas
        if states[0] == 0 and actions[0] == 1:
            pass #import ipdb; ipdb.set_trace()
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

Q = np.zeros((grid_width**2, 4))
# hack to make final backups work properly:
Q[-1] = np.ones(4)
pi = EpsilonGreedy(eps / 10.)
mu = EpsilonGreedy(eps)
policy = mu

returns = []

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
        if step_n > backup_length: # forward view...
            estimated_return = estimate_return(mu, pi, Q, states[-backup_length:], actions[-backup_length:])#, sigmas=np.zeros(backup_length))
            #print estimated_return
            s_up = states[-backup_length]
            a_up = actions[-backup_length]
            Q[s_up, a_up] = (1 - lr) * Q[s_up, a_up] + lr * estimated_return
        if s == grid_width**2 -1:
            # TODO: the final backups!! (without them, nothing ever improves :P)
            for extra_step in range(backup_length):
                step_n += 1
                states.append(s)
                actions.append(rng.choice(4))
                estimated_return = estimate_return(mu, pi, Q, states[step_n - backup_length: step_n], actions[step_n - backup_length: step_n])#, sigmas=np.zeros(backup_length))
                #print estimated_return
                Q[s,a] = (1 - lr) * Q[s,a] + lr * estimated_return
            finished = 1
        #print states
        #print actions
        #import ipdb; ipdb.set_trace()

    # returns
    print episode, len(states) - backup_length
    returns.append(-len(states) + backup_length)





