
import copy
from feature_extractor import FeatureTrueState

# TODO: sample a different number of rewards for different (s,a) pairs
#def sample_rewards(agent, queries, sampled_R):
#   """ queries[s,a] = # of queries to perform on (s,a) """
#   return {(s,a) : np.random.normal(sampled_R[s,a], 1, queries[s,a]) for (s,a) in agent.P_prior.keys()}

# TODO: maintaining a posterior over the variance (or maybe just using Bernoullis?) should help a lot!

# TODO: optionally normalize rewards (or performance??) across different sampled_R
# TODO: docstrings

def estimate_performance_ASQR(agent, sampled_R, sampled_rewards, query_cost):
    """
    sampled_R: a dictionary of mean rewards for each (s,a) pair, sampled from the agent's posterior.
    sampled_rewards: a dictionary of (several) sampled rewards for each (s,a) pair, sampled from sampled_R
    """
    _, P_hat = agent.map_mdp()
    R_hat = agent.R_prior
    updated_R = {}
    for [s,a] in sampled_rewards:
        mu0, tau0 = R_hat[s,a]
        num_samples = len(sampled_rewards[s,a])
        tau1 = tau0 + agent.tau * num_samples
        mu1 = (mu0 * tau0 + sum(sampled_rewards[s,a]) * agent.tau) / tau1
        updated_R[s,a] = mu1
    return agent.compute_qVals_true(updated_R, P_hat, sampled_R, P_hat)[0] - query_cost * sum([len(qq) for qq in sampled_rewards.values()])
    #return compute_qVals_true(agent, updated_R, P_hat, sampled_R, P_hat)[0] - query_cost * sum([len(qq) for qq in sampled_rewards.values()])

# We can run this in the loop every so often...
def run_ASQR(agent, n_max, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    Use ASQR to select an n between 0 and n_max (inclusive).
    """
    performances = []
    for k in range(num_R_samples):
        sampled_R = agent.sample_mdp()[0]
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], 1, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        performances.append([estimate_performance_ASQR(agent, sampled_R, first_n_sampled_rewards[n], query_cost) for n in range(n_max+1)])
    avg_performances = np.array(performances).sum(0)
    return avg_performances.argmax()

def run_SQR(agent, n_max, env, epLen, query_cost=1., num_R_samples=1, normalize_rewards=False):
    """
    Use SQR to select an n between 0 and n_max (inclusive).
    """
    performances = []
    for k in range(num_R_samples):
        sampled_R = agent.sample_mdp()[0]
        sampled_rewards = {(s,a) : np.random.normal(sampled_R[s,a], 1, n_max) for (s,a) in agent.P_prior.keys()}
        first_n_sampled_rewards = [{sa: sampled_rewards[sa][:n] for sa in sampled_rewards} for n in range(n_max + 1)]
        # make copies for run_finite_tabular_experiment
        agent_copy = copy.deepcopy(agent)
        env_copy = copy.deepcopy(env)
        env_copy.R = {kk:(sampled_R[kk], 1) for kk in sampled_R}
        env_copy.P = agent.P_prior
        f_ext = FeatureTrueState()
        performances.append([run_finite_tabular_experiment(agent_copy, env_copy, f_ext, epLen, sampled_rewards=first_n_sampled_rewards[n]) for n in range(n_max+1)])
    avg_performances = np.array(performances).sum(0)
    return avg_performances.argmax()



#---------------------------------------------------
# CODE FOR LATER
# Code that does Bayesian updating for a Gaussian (?)
# from: http://engineering.richrelevance.com/bayesian-analysis-of-normal-distributions-with-python/
from numpy import sum, mean, size, sqrt
from scipy.stats import norm, invgamma

def draw_mus_and_sigmas(data,m0,k0,s_sq0,v0,n_samples=10000):
    # number of samples
    N = size(data)
    # find the mean of the data
    the_mean = mean(data) 
    # sum of squared differences between data and mean
    SSD = sum( (data - the_mean)**2 ) 

    # combining the prior with the data - page 79 of Gelman et al.
    # to make sense of this note that 
    # inv-chi-sq(v,s^2) = inv-gamma(v/2,(v*s^2)/2)
    kN = float(k0 + N)
    mN = (k0/kN)*m0 + (N/kN)*the_mean
    vN = v0 + N
    vN_times_s_sqN = v0*s_sq0 + SSD + (N*k0*(m0-the_mean)**2)/kN

    # 1) draw the variances from an inverse gamma 
    # (params: alpha, beta)
    alpha = vN/2
    beta = vN_times_s_sqN/2
    # thanks to wikipedia, we know that:
    # if X ~ inv-gamma(a,1) then b*X ~ inv-gamma(a,b)
    sig_sq_samples = beta*invgamma.rvs(alpha,size=n_samples)

    # 2) draw means from a normal conditioned on the drawn sigmas
    # (params: mean_norm, var_norm)
    mean_norm = mN
    var_norm = sqrt(sig_sq_samples/kN)
    mu_samples = norm.rvs(mean_norm,scale=var_norm,size=n_samples)

    # 3) return the mu_samples and sig_sq_samples
    return mu_samples, sig_sq_samples





