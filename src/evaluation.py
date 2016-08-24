from pylab import *
load_str = 'junk.txt'
def plot_regret(load_str=load_str):
    eps = []
    cum_regrets = []
    perfs = []
    for line in open(load_str).readlines()[1:]:
        ep, _, _, cum_regret, _, perf  = line.split(',')
        eps.append(ep)
        cum_regrets.append(cum_regret)
        perfs.append(perf)
    figure()
    plot(eps, cum_regrets)
    figure()
    plot(eps, perfs)


def compute_performance(cum_reward, perf, query_cost_multiplier):
    query_cost = cum_reward - perf
    return cum_reward - query_cost * query_cost_multiplier


def plot_performance(load_str, query_cost=None):
    filename = load_str.split('/')[-1]
    orig_query_cost = float(filename.split('__')[0].split("=")[1])
    if query_cost is None:
        query_cost = orig_query_cost
    eps = []
    cum_rewards = []
    orig_perfs = []
    perfs = []
    orig_query_costs = []
    for line in open(load_str).readlines()[1:]:
        ep, _, cum_reward, _, _, perf  = line.split(',')
        eps.append(int(ep))
        cum_rewards.append(float(cum_reward))
        orig_perfs.append(float(perf))
        orig_query_costs.append(orig_perfs[-1] - cum_rewards[-1])
        perfs.append(cum_rewards[-1] + orig_query_costs[-1] * query_cost / orig_query_cost)
    #figure()
    plot(eps, perfs)
    return perfs
    

#----------------------------------------------------
# Below: for dk_exp_script (Aug 24)

"""
TODO: make a plotting function that does all of the below (so after I run experiments, I just pass in the results and get what I want...)

TODO: figure out what plots to make...
    we can also average over one axis...

    And there are other things we want to look at, too...
        e.g. how well does ASQR / SQR perform?
            e.g. integral of P_SQR(n) * performance(n)   (...for a given c, h)

"""

def compute_performance(returns, num_queries, query_cost):
    return returns - num_queries * query_cost

# these will be 2d arrays of shape (log_num_episodes, log_n_max)
avg_returns= np.load(save_str + 'returns.npy').mean(axis=0)
avg_num_queries= np.load(save_str + 'num_queries.npy').mean(axis=0)
query_costs = 2**np.linspace(-10,10,21)
# TODO: construct this 3d array
avg_perf = [ [ [compute_performance(returns[n], num_queries[n], query_cost) for n in range(len(returns))] 
              for (returns, num_queries) in zip(avg_returns, avg_num_queries) ]
            for query_cost in query_costs]
avg_perf = np.array(avg_perf)

# TODO: label axes
def fixed_n_plots(avg_perf):
    """avg_perf is a 3d array, indexed by cost, horizon, n"""

    print "avg_perf.shape=", avg_perf.shape
    print "N.B. all of the of the values are indices in log-scales"

    figure()
    suptitle('performance curves for best values of c, h, n')
    best_c = np.argmax(np.mean(avg_perf, (1,2)))
    best_h = np.argmax(np.mean(avg_perf, (0,2)))
    best_n = np.argmax(np.mean(avg_perf, (0,1)))
    print 'performance curves for best values of c, h, n'
    print "indices of best values:   ", best_c, best_h, best_n
    # plot performance as a function of c, h, n:
    subplot(2,3,1)
    plot(avg_perf[:,best_h, best_n])
    subplot(2,3,2)
    plot(avg_perf[best_c, :, best_n])
    subplot(2,3,3)
    plot(avg_perf[best_c, best_h, :])
    # plot performance heat-maps with fixed c, h, n:
    subplot(2,3,4)
    imshow(avg_perf[best_c, :, :], cmap="Greys")
    subplot(2,3,5)
    imshow(avg_perf[:, best_h, :], cmap="Greys")
    subplot(2,3,6)
    imshow(avg_perf[:, :, best_n], cmap="Greys")

    # TODO: plot each of these things for some different values of all of the others... (does taking the "best" even make sense??)
    figure()
    suptitle('performance heatmaps for all values of c')
    for c_ind in range(avg_perf.shape[0]):
        nrows = int(avg_perf.shape[0]**.5) + 1
        subplot(nrows, nrows, c_ind+1)
        imshow(avg_perf[c_ind], cmap="Greys")
    figure()
    suptitle('performance heatmaps for all values of h')
    for ind in range(avg_perf.shape[1]):
        nrows = int(avg_perf.shape[1]**.5) + 1
        subplot(nrows, nrows, ind+1)
        imshow(avg_perf[:,ind], cmap="Greys")
    figure()
    suptitle('performance heatmaps for all values of n')
    for ind in range(avg_perf.shape[2]):
        nrows = int(avg_perf.shape[2]**.5) + 1
        subplot(nrows, nrows, ind+1)
        imshow(avg_perf[:,:,ind], cmap="Greys")

fixed_n_plots(avg_perf)









