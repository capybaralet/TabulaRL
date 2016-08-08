
def sample_R(agent):
    return agent.sample_mdp()[0]

sampled_R = sample_R(agent)

def sampled_rewards(agent, queries, sampled_R):
    """ queries[s,a] = # of queries to perform on (s,a) """
    # draw samples from R_hat
    samples = {(s,a) : np.random.gaussian(sampled_R[s,a][0], sampled_R[s,a][1], queries[s,a])}
    return samples

# TODO: we could also try sampling the rewards independently for each value of n 
#        (this would make the comparison btw different ns more stochastic)
def estimate_performance(agent, sampled_rewards, query_cost, sampled_R):
    """ we pass the first n sampled_rewards from the function above"""
    R_hat, P_hat = agent.map_mdp()
    updated_R = {}
    for [s,a] in sampled_rewards:
        mu0, tau0 = R_hat[s,a]
        num_samples = len(sampled_rewards[s,a])
        tau1 = tau0 + self.tau * num_samples
        mu1 = (mu0 * tau0 + reward * self.tau * num_samples) / tau1
        updated_R[s,a] = mu1, tau1
    return max(agent.compute_qVals(updated_R, P_hat)[0]) - query_cost * sum(queries.values())










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





