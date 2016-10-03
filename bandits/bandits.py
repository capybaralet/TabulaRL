"""
Library for active Multi-armed bandits

Written by Jan Leike 2016


Conventions:

T: number of times pulled each arm
s: number of successes on each arm
n: horizon
t: current time step
"""

import random
from math import sqrt, ceil
import numpy as np
from scipy.stats import beta


def bestArm(T, s):
    """Return the best arm (greedy policy)"""
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    return np.argmax(mu_hats)


def EpsGreedyPolicy(T, s, n, t, d=0.1, c=0.05):
    k = len(T)  # number of arms
    eps_t = c*k / d**2 / t  # from the theory, d is the minimum gap in the arms
#    eps_t = 1/t
    if random.random() > eps_t:
        # greedy policy
        return bestArm(T, s)
    else:
        # pick an action uniformly at random
        return random.randint(0, k - 1)


def ThompsonPolicy(T, s, n, t):
    """policy based on Thompson sampling"""
    mu_hats = beta.rvs(np.add(s, 1), np.subtract(np.add(T, 1), s))
    return np.argmax(mu_hats)


def UCBPolicy(T, s, n, t, c=0.5):
    """policy based on the UCB algorithm"""
    if 0 in T:
        return T.index(0)
    else:
        ucb = np.array(s) / np.array(T) + np.sqrt(c * np.log(t) / np.array(T))
        return max(ucb, key=lambda x: (x, random.random()))


def OCUCBPolicy(T, s, n, t, c=0.5):
    """Policy based on the optimally confident UCB algorithm
    by Lattimore (2015)"""
    if 0 in T:
        return T.index(0)
    else:
        ucb = np.array(s) / np.array(T) + \
              np.sqrt(c * np.log(n / float(t)) / np.array(T))
        return max(ucb, key=lambda x: (x, random.random()))


def BayesUCBPolicy(T, s, n, t):
    """Bayes-UCB policy with quartile 1/t"""
    if 0 in T:
        return T.index(0)
    else:
        a = np.add(s, 1)
        b = np.subtract(np.add(T, 1), s)
        quantiles = beta.isf(1 / float(t), a, b)
        return max(quantiles, key=lambda x: (x, random.random()))


def Arm1Policy(T, s, n, t, cost=0):
    """policy that always pulls arm 1"""
    return 0, False


def RoundRobinPolicy(T, s, n, t, cost=0):
    """policy that alternates between all arms"""
    return t % len(T)

# function [j query] = activeBanditPolicy1(T, s, n, t, cost, c = 1/2)
# 	% shitty heuristic
# 	k = length(T); % number of arms
# 	if any(T == 0)
# 		j = find(T == 0, 1);
# 		query = true;
# 	else
# 		ucb = s ./ T + sqrt(c * log(n / t) ./ T) - cost * T / (n - t + 1);
# 		m = max(ucb);
# 		mu_hats = (s + 1) ./ (T + 2);
# 		idx = find(ucb == m);
# 		if m <= max(mu_hats)
# 			[_, j] = max(mu_hats);
# 			query = false;
# 		else
# 			j = idx(1 + floor(length(idx) * rand()));
# 			query = true;
# 		end
# 	end
# end
#
# function [j query] = activeBanditPolicy2(T, s, n, t, cost, c = 1/2)
# 	% shitty heuristic
# 	k = length(T); % number of arms
# 	if any(T == 0)
# 		j = find(T == 0, 1);
# 		query = true;
# 	else
# 		ucb = s ./ T + sqrt(c * log(n / t) ./ T) - cost * sqrt(T / n);
# 		m = max(ucb);
# 		mu_hats = (s + 1) ./ (T + 2);
# 		idx = find(ucb == m);
# 		j = idx(1 + floor(length(idx) * rand()));
# 		idx2 = 1:k;
# 		idx2([j]) = [];
# 		if m > max(mu_hats(idx2)) || T(j) < max(T(idx2))
# 			query = true;
# 		else
# 			[_, j] = max(mu_hats);
# 			query = false;
# 		end
# 	end
# end


def expectedRegret(T, s, n, t, arm, tol=1e-3):
    """Bayes-expected regret when committing to the arm 'arm'"""
    k = len(T)
    a = np.add(s, 1)
    b = np.subtract(np.add(T, 1), s)

    def f(x, j):
        """integrant for E [ theta_j - theta_arm | j is best arm ]"""
        assert j != arm
        y = x * beta.cdf(x, a[arm], b[arm])
        y -= beta.expect(lambda z: z, (a[arm], b[arm]), lb=0, ub=x)
        for s in range(k):
            if s != arm and s != j:
                y *= beta.cdf(x, a[s], b[s])
        return y

    x = 0
    for j in range(k):
        if j != arm:
            x += beta.expect(lambda z: f(z, j), (a[j], b[j]))
    return x * (n - t)


def probBestArm(T, s, arm, tol=1e-3):
    """posterior probability that "arm" is the best arm"""
    k = len(T)
    a = np.add(s, 1)
    b = np.subtract(np.add(T, 1), s)

    def f(x):
        y = 1
        for i in range(k):
            if s != arm:
                y *= beta.cdf(x, a[i], b[i])
    return beta.expect(f, (a[arm], b[arm]))


def minExpectedRegret(T, s, n, t, tol=1e-3):
    """Bayes-expected regret when committing to the best arm"""
    return expectedRegret(T, s, n, t, bestArm(T, s), tol)


def DMEDPolicy(T, s, n, t, cost):
    # Honda, Junya, and Akimichi Takemura.
    # An Asymptotically Optimal Bandit Algorithm for Bounded Support Models.
    # COLT 2010.
    # here we use (theta - theta^*) instead of KL(B(theta), B(theta^*))
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    if 0 in T:
        arms = range(k)  # prevent log(0) errors
    else:
        Jp = np.multiply(T, np.subtract(m, mu_hats)) - np.log(n) + np.log(T)
        arms = []
        for i in range(k):
            if Jp[i] <= 0:
                arms.append(i)
    return random.choice(arms), len(arms) > 1


# def nonlinearSequentialElim(T, s, n, t, cost, p = 1.5):
#     # From https://arxiv.org/abs/1609.02606
#     assert p > 0
#     k = len(T)
#     budget = n**(2/3.)
#     C_p = sum([float(x)**(-p) for x in [2] + range(2, k + 1)])
#     n = ceil((budget - k) / C_p * (k - (0:(k-2))) .^ (-p));
#     # ...


def FixedQueryPolicy(T, s, n, t, cost, query_for=float('inf'),
                     alg=OCUCBPolicy):
    """use standard bandit algorithm and query the query_for steps"""
    if t <= query_for:
        return alg(T, s, n, t), True
    else:
        return bestArm(T, s), False


def EpsQueryPolicy(T, s, n, t, cost, alg=OCUCBPolicy):
    """use OCUCB and query with probability 1/t"""
    if random.random() < 1/t:
        return alg(T, s, n, t), True
    else:
        return bestArm(T, s), False


def ExpQueryPolicy(T, s, n, t, cost, alg=OCUCBPolicy):
    """query whenever doing an exploration action"""
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    j = alg(T, s, n, t)
    if mu_hats[j] == max(mu_hats):
        return j, T[j] < max(T)
    else:
        return j, True

# function s = querySteps(T, s)
#     % the number of steps you expect to need to
#     % bring the two arms with the highest means together
#     % (crude heuristic)
#     k = length(T);
#     mu_hats = (s + 1) ./ (T + 2);
#     [m j] = max(mu_hats);
#     min_gap_steps = min(((T + 1) .* (m - mu_hats))([(1:(j-1)) ((j+1):k)]));
#     s = 2*ceil(min_gap_steps + 0.01)^2;
# end


def querySteps3(T, s):
    """
    the number of steps you expect to need to
    bring the two arms with the highest means together
    note: this code is horribly inefficient, but that should't matter ^_^
    """
    def mu_hats(T, s):
        return np.add(s, 1) / np.add(T, 2).astype(float)
    mu_hats_ = list(mu_hats(T, s))
    j = np.argmax(mu_hats_)
    del mu_hats_[j]
    i = np.argmax(mu_hats_)
    i += 1 if i >= j else 0
    assert i != j
    z = int(ceil(sqrt(2*(min(T[i], T[j]) + 2))))  # upper bound
    l = [(xi, xj) for xi in range(z + 1) for xj in range(z + 1)]
    l.sort(key=lambda (xi, xj): xi**2 + xj**2)
    for (xi, xj) in l:
        if mu_hats(T[i] + 2*xi**2, s[i] + xi + 2*xi**2*mu_hats[i]) >= \
           mu_hats(T[j] + 2*xj**2, s[j] - xj + 2*xj**2*mu_hats[j]):
            return 2*(xi**2 + xj**2)
    return float('inf')


def DMED(T, s, n, t):
    """DMED bandit policy"""
    k = len(T)
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    m = max(mu_hats)
    if 0 in T:
        arms = range(k)  # prevent log(0) errors
    else:
        Jp = np.multiply(T, np.subtract(m, mu_hats)) - np.log(n) + np.log(T)
        arms = []
        for i in range(k):
            if Jp[i] <= 0:
                arms.append(i)
    return random.choice(arms)


def parameterizedRegretQuery(T, s, n, t, cost, banditpolicy=DMED, alpha=0.35):
    """
    execute bandit policy until
    cost to move posterior < alpha * expected regret
    with parameter alpha \in (0, 1)
    """
    mu_hats = np.add(s, 1) / np.add(T, 2).astype(float)
    best_arm = np.argmax(mu_hats)
    query_steps = querySteps3(T, s)
    if t + query_steps >= n:
        # instant commitment because the time frame is too long
        return best_arm, False
    query = cost * query_steps < alpha * minExpectedRegret(T, s, n, t)
    if query:
        return banditpolicy(T, s, n, t), True
    else:
        return best_arm, False


def playBernoulli(mu, n, cost, policy, args):
    """Play a game of bernoulli arms

    mu: list of Bernoulli parameters
    n: horizon
    policy: the policy to use
    cost: the query cost
    """
    k = len(mu)  # number of arms
    T = [0]*k  # number of times pulled each arm
    s = [0]*k  # number of successes on each arm
    regret = 0  # cumulative undiscounted regret
    cregret = [0]
    mu_best = max(mu)
    query = True
    for t in range(n):
        old_query = query
        j, query = policy(T, s, n, t, cost, *args)

        # Pull arm j
        r = int(mu[j] > random.random())  # reward = 0 or 1
        if query:
            T[j] += 1
            s[j] += r
            regret += cost
        elif old_query:
            print('stopping at t = %d. Commited to arm %d' % (t, j + 1))
        regret += mu_best - mu[j]
        cregret.append(regret)
    print('regret = %.2f' % regret)
    return cregret


def runExperiment(mu, n, cost, policy, N, args):
    cregrets = []
    for i in range(N):
        cregrets.append(playBernoulli(mu, n, cost, policy, args))
    return cregrets
