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

# TODO: plotting for SQR/ASQR
#   e.g. integral of P_SQR(n) * performance(n)   (...for a given c, h)
# TODO: normalized performance

# TODO: plot performance / tstep


def multiplot(list_of_curves, transpose=True):
    if transpose:
        list_of_curves = list_of_curves.T
    num_plots = len(list_of_curves) 
    colormap = plt.cm.gist_ncar
    gca().set_color_cycle([colormap(i) for i in np.linspace(0.05, 0.95, num_plots)])
    for curve in list_of_curves:
        plot(curve)

def compute_performance(returns, num_queries, query_cost):
    return returns - num_queries * query_cost

# these will be 2d arrays of shape (log_num_episodes, log_n_max)
avg_returns= np.load(save_str + 'returns.npy').mean(axis=0)
avg_num_queries= np.load(save_str + 'num_queries.npy').mean(axis=0)
query_costs = 2**np.linspace(-10,10,21)
avg_perf = [ [ [compute_performance(returns[n], num_queries[n], query_cost) for n in range(len(returns))] 
              for (returns, num_queries) in zip(avg_returns, avg_num_queries) ]
            for query_cost in query_costs]
avg_perf = np.array(avg_perf)

# TODO: label axes
# TODO: plotting function
def fixed_n_plots(avg_perf):
    """avg_perf is a 3d array, indexed by cost, horizon, max_num_queries"""

    print "avg_perf.shape=", avg_perf.shape
    print "N.B. all of the of the values are indices in log-scales"
    print "\n blue is low, pink is high"

    figure()
    suptitle('performance curves for best values of c, h, n')
    best_c = np.argmax(np.mean(avg_perf, (1,2)))
    best_h = np.argmax(np.mean(avg_perf, (0,2)))
    best_n = np.argmax(np.mean(avg_perf, (0,1)))
    print 'performance curves for best values of c, h, n'
    print "indices of best values:   ", best_c, best_h, best_n
    if 0:
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

    # TODO: emphasize contrast in these plots (e.g. using softmax with high temperature???)
    if 0:
        figure()
        suptitle('performance heatmaps for all values of c; x=h, y=n')
        for ind in range(avg_perf.shape[0]):
            nrows = int(avg_perf.shape[0]**.5) + 1
            subplot(nrows, nrows, ind+1)
            imshow(avg_perf[ind], cmap="Greys")
            #imshow(softmaxx(avg_perf[ind], temp=1e6), cmap="Greys")
        figure()
        suptitle('performance heatmaps for all values of h; x=c, y=n')
        for ind in range(avg_perf.shape[1]):
            nrows = int(avg_perf.shape[1]**.5) + 1
            subplot(nrows, nrows, ind+1)
            imshow(avg_perf[:,ind], cmap="Greys")
        figure()
        suptitle('performance heatmaps for all values of n; x=c, y=h')
        for ind in range(avg_perf.shape[2]):
            nrows = int(avg_perf.shape[2]**.5) + 1
            subplot(nrows, nrows, ind+1)
            imshow(avg_perf[:,:,ind], cmap="Greys")

    # more plots
    if 0:
        figure()
        suptitle('performance curves for all values of c; x=h, y=perf, curves: n')
        for ind in range(0, avg_perf.shape[0]):
            nrows = int(avg_perf.shape[0]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[ind])
            scale = np.max(avg_perf[ind])
            ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of c; x=n, y=perf, curves:h')
        for ind in range(avg_perf.shape[0]):
            nrows = int(avg_perf.shape[0]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[ind].T)
            scale = np.max(avg_perf[ind])
            ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of h; x=c, y=perf, curves: n')
        for ind in range(avg_perf.shape[1]):
            nrows = int(avg_perf.shape[1]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[:, ind])
            scale = np.max(avg_perf[:, ind])
            ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of h; x=n, y=perf, curves:c')
        for ind in range(avg_perf.shape[1]):
            nrows = int(avg_perf.shape[1]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[:, ind].T)
            scale = np.max(avg_perf[:, ind])
            ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of n; x=c, y=perf, curves: h')
        for ind in range(avg_perf.shape[2]):
            nrows = int(avg_perf.shape[2]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[:, :, ind])
            scale = np.max(avg_perf[:, :, ind])
            ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of n; x=h, y=perf, curves:c')
        for ind in range(avg_perf.shape[2]):
            nrows = int(avg_perf.shape[2]**.5) + 1
            subplot(nrows, nrows, ind+1)
            multiplot(avg_perf[:, :, ind].T)
            scale = np.max(avg_perf[:, :, ind])
            ylim(-.4*scale, 1.1*scale)

    # even more plots
    if 1:
        figure()
        suptitle('performance curves for all values of c; x=h, y=perf, curves: n')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[0], avg_perf.shape[0] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[ind])
                scale = np.max(avg_perf[ind])
                ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of c; x=n, y=perf, curves:h')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[0], avg_perf.shape[0] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[ind].T)
                scale = np.max(avg_perf[ind])
                ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of h; x=c, y=perf, curves: n')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[1], avg_perf.shape[1] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[:, ind])
                scale = np.max(avg_perf[:, ind])
                ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of h; x=n, y=perf, curves:c')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[1], avg_perf.shape[1] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[:, ind].T)
                scale = np.max(avg_perf[:, ind])
                ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of n; x=c, y=perf, curves: h')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[2], avg_perf.shape[2] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[:, :, ind])
                scale = np.max(avg_perf[:, :, ind])
                ylim(-.4*scale, 1.1*scale)
        figure()
        suptitle('performance curves for all values of n; x=h, y=perf, curves:c')
        for plot_n, ind in enumerate(range(0, avg_perf.shape[2], avg_perf.shape[2] / 4)):
            if plot_n < 4:
                print ind
                nrows = 2
                subplot(nrows, nrows, plot_n+1)
                multiplot(avg_perf[:, :, ind].T)
                scale = np.max(avg_perf[:, :, ind])
                ylim(-.4*scale, 1.1*scale)

fixed_n_plots(avg_perf)









