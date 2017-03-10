"""
WIP


BASIC SETUP
------------
We look at n-step TD control, using the Q(sigma) algorithm from Sutton and Barto.

The environment is a deterministic gridworld with -1 reward everywhere.
The agent starts in the top-left corner, and the bottom-right is terminal, so we don't need any reward discounting.

We use epsilon greedy; our target policy has epsilon=0.


IMPLEMENTATION
--------------
We record each episode and update between episodes, based on the last trajectory, using ALL of the partial trajectories.


DETAILS
--------------
rollout: [states, actions]
    we don't need to track rewards; they are all the same!

Q: 2d array of shape (s, a)

policy: the BEHAVIOUR policy (TARGET is greedy!)


"""

import numpy
np = numpy
import numpy.random

# from http://www.iro.umontreal.ca/~memisevr/code/logreg.py
def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels.
    """
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape, dtype="int")
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result

#----------------------------------------
# hparams

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backup_steps', type=int, default=3)
parser.add_argument('--eps', type=float, default=.1)
parser.add_argument('--grid_width', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4) # learning rate
parser.add_argument('--num_episodes', type=int, default=50000)
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

if backup_steps is None:
    backup_steps = grid_width

# RANDOM SEED
if seed is not None:
    np.random.seed(seed)  # for reproducibility
    rng = numpy.random.RandomState(seed)
else:
    rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

#----------------------------------------
# policy 
class EpsilonGreedy(object):
    def __init__(self, eps):
        self.eps = eps

    def P_a(self, Q_vals):
        return self.eps * np.ones(4) / 4. + (1-self.eps) * onehot(np.argmax(Q_vals))

    def sample(self, Q_vals):
        if rng.rand() > self.eps:
            return np.argmax(Q_vals)
        else:
            return rng.choice(len(Q_vals))


#----------------------------------------
# environment
def grid_episode(grid_width, policy, Q):
    """
    states are ordered lexigraphically
    actions are ordered "clockwise" (starting from north = 0)
    """
    states = []
    actions = []
    rewards = []
    s = 0
    while True:
        a = policy.sample(Q[s])
        states.append(s)
        actions.append(a)
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
        if s == grid_width**2 -1:
            return [np.array(states), np.array(actions)]


#----------------------------------------
# ...


#def update_Q(policy, Q, rollout):
def get_updates(mu, pi, Q, rollout):
    states, actions = rollout
    num_tsteps = len(states)
    tsteps = np.arange(num_tsteps)

    # action probabilities
    mu_P_a = [mu.P_a(Q[s]) for s in rollout[0]]
    pi_P_a = [pi.P_a(Q[s]) for s in rollout[0]]

    # importance sampling ratios weights (either 0 or 1/P_a)
    rho_t = [(greedy_actions[t] == a) / (1 - .75 * eps) for t,a in enumerate(rollout[1])]
    #
    #rho_t = [policy(Q[s])[a] for (s,a) in zip(rollout[0], rollout[1])]

    # COMPUTE TRUNCATED RETURNS (TODO: use Q to get SARSA-n!)
    # truncate returns
    estimated_returns = tsteps * (tsteps < backup_steps) + backup_steps * (tsteps >= backup_steps)
    # reverse and negate returns
    estimated_returns = -1 * estimated_returns[::-1]

    updates = np.zeros((grid_width**2, 4))
    visit_counts = np.zeros((grid_width**2, 4))
    for tstep, (s, a) in enumerate(zip(states, actions)):
        visit_counts[s,a] += 1
        # TODO: use Q-sigma
        updates[s,a] = estimated_returns[tstep]
    
    return updates, visit_counts


# TODO
def estimated_returns(rho, sigma, Q_vals, rollout):
    states, actions = rollout
    




#----------------------------------------
# RUN THINGS

Q = np.zeros((grid_width**2, 4))
# TODO
#pi = EpsilonGreedy(eps / 10.)
mu = EpsilonGreedy(eps)
policy = mu

returns = []

for episode in range(num_episodes):
    #lr *= .9999
    rollout = grid_episode(grid_width, policy, Q)
    print episode, len(rollout[0])
    returns.append(-len(rollout[0]))
    # TODO: monitoring
    updates, visit_counts = get_updates(policy, Q, rollout)
    # will this work??? this is like "every-visit" Q-learning 
    Q = (1 - lr * visit_counts) * Q + lr * updates
    #Q = update_Q(policy, Q, rollout)







