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
    

def compute_performance(returns, num_queries, query_cost):
    return returns - num_queries * query_cost

# TODO: construct this 3d array
query_costs = 
avg_perf = [ [compute_performance(returnz, num_queriez, query_cost) for n in range(len(returns))] for query_cost in query_costs]

def fixed_n_plots(avg_perf):
    """avg_perf is a 3d array, indexed by cost, horizon, n"""
    figure()
    suptitle('performance curves')
    best_c = np.argmmax(avg_perf, 0)
    best_h = np.argmmax(avg_perf, 1)
    best_n = np.argmmax(avg_perf, 2)
    # plot performance as a function of c, h, n:
    subplot(4,3,1)
    plot(avg_perf[:,best_h, best_n])
    subplot(4,3,2)
    plot(avg_perf[best_c, :, best_n])
    subplot(4,3,3)
    plot(avg_perf[best_c, best_h, :])
    # plot performance heat-maps with fixed c, h, n:
    # imshow
    subplot(4,3,4)
    imshow(avg_perf[best_c, :, :])
    subplot(4,3,5)
    imshow(avg_perf[:, best_h, :])
    subplot(4,3,6)
    imshow(avg_perf[:, :, best_n])
    # TODO: level curves
    #figure()
    #suptitle('level curves ("fixed" performance) ')

    

"""
TODO: make a plotting function that does all of the below (so after I run experiments, I just pass in the results and get what I want...)

TODO: figure out what plots to make...
    Performance is a scalar function of R^3: performance(n, c, h)
        so I should pass a 3d array of expected performance for different values
    So we either: 
        plot performance vs. _______
        make level curves
            ...but we can't compute it... :/
    and we either:
        make a line plot
        make a heat-map

    In any case, we need to figure out which values to look at...
        HOW?

    And there are other things we want to look at, too...
        e.g. how well does ASQR / SQR perform?
            e.g. integral of P_SQR(n) * performance(n)   (...for a given c, h)

"""









