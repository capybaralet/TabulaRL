from pylab import *
import os

#----------------------------------------------------
# Below: for dk_exp_script (Aug 24)
# TODO: normalized performance

# COMPUTE PERFORMANCE FROM RESULTS FILES
def compute_performance(returns, num_queries, query_cost):
    perf = returns - num_queries * query_cost
    return perf


# FIXME: check that results make sense (larger cost gives higher performance!?!?!??!)
def compute_performances(save_str, which_slice=None, query_costs=2**np.linspace(-10,10,21) ):
    """returns avg_perf(_per_tstep):  a 3d array, indexed by cost (c), num_episodes (h), max_num_queries (n)"""
    # these will be 2d arrays of shape (log_num_episodes, log_n_max)
    returns= np.load(os.path.join(save_str, 'returns.npy'))
    num_queries= np.load(os.path.join(save_str, 'num_queries.npy'))
    if which_slice is None:
        which_slice = range(len(returns))
    avg_returns= returns[which_slice].mean(axis=0)
    avg_num_queries= num_queries[which_slice].mean(axis=0)
    avg_perf = [ [ [compute_performance(returns[n], num_queries[n], query_cost) for n in range(len(returns))] 
                  for (returns, num_queries) in zip(avg_returns, avg_num_queries) ]
                for query_cost in query_costs]
    avg_perf = np.array(avg_perf)
    avg_perf_per_tstep = avg_perf / 2**np.arange(avg_perf.shape[1]).reshape((1, avg_perf.shape[1], 1))
    return avg_perf, avg_perf_per_tstep

log_n_max = 10
ns = np.hstack((np.array([0,]), 2**np.arange(log_n_max)))

# TODO: figure out how to plot query_functions with multiple params 
def plot_performance(avg_perf, horizons, query_costs, label='', ns=ns, title_extra=''):
    """ Makes the nice figure that we'd like to see... """
    suptitle(title_extra + ': (Estimated) average performance per timestep (y) as a function of max number of queries (x)')
    #horizons = ['short', 'long']
    #query_costs = ['small', 'medium', 'large']
    horizons = ['32', '1024']
    query_costs = ['.002', '.125', '1.']
    for row, horizon in enumerate(horizons):
        for column, query_cost in enumerate(query_costs):
            print column + row*len(query_costs) + 1
            subplot(len(horizons), len(query_costs), column + row*len(query_costs) + 1)
            if column == 0:
                ylabel('horizon='+horizon)
            if row == 0:
                title('query_cost='+query_cost)
            else:
                xlabel('max # queries per state action')
            plot(avg_perf[column, row], label=label)
            if ns is not None:
                xticks(range(len(ns)), ns)



#------------------------------------
# ACTUALLY MAKE PLOTS

# chain5 (PSRL vs. PSRLLimited...)
save_strs = [
        '2016-08-25_18:02:44___agent=PSRLLimitedQuery__algorithm=fixed_n__environment=chain5__log_n_max=10__log_num_episodes=10__normalize_rewards=0__num_R_samples=3000__query_cost=1.0/',
        '2016-08-25_18:02:44___agent=PSRL__algorithm=fixed_n__environment=chain5__log_n_max=10__log_num_episodes=10__normalize_rewards=0__num_R_samples=3000__query_cost=1.0'
]
save_strs = ['results/results__dk_exp_script.py/' + ss for ss in save_strs]

figure()
for label, save_str in zip(['PSRL_clamped', 'PSRL_continue_sampling'], save_strs):
    avg_perf, avg_perf_per_tstep = compute_performances(save_str)
    horizons = [5, 10]
    query_costs = [1,7,10]
    representative_perfs = np.empty((3, 2, avg_perf.shape[2]))
    for ii, qq in enumerate(query_costs):
        for jj, hh in enumerate(horizons):
            representative_perfs[ii, jj] = avg_perf_per_tstep[qq,hh]
    horizons = [str(hh) for hh in horizons]
    query_costs = [str(qq) for qq in query_costs]
    plot_performance(representative_perfs, label=label, title_extra='chain5', horizons=horizons, query_costs=query_costs)
    legend(loc=4)

# chain5
save_strs = [
        '2016-08-25_18:02:44___agent=PSRLLimitedQuery__algorithm=fixed_n__environment=chain5__log_n_max=10__log_num_episodes=10__normalize_rewards=0__num_R_samples=3000__query_cost=1.0/',
        '2016-08-25_18:02:42___agent=PSRLLimitedQuery__algorithm=SQR__environment=chain5__log_n_max=10__log_num_episodes=10__normalize_rewards=0__num_R_samples=3000__query_cost=1.0/',
        '2016-08-28_01:10:13___agent=PSRLLimitedQuery__algorithm=ASQR__environment=chain5__log_n_max=10__log_num_episodes=10__num_R_samples=3000/'
]
save_strs = ['results/results__dk_exp_script.py/' + ss for ss in save_strs]

figure()
for label, save_str in zip(['fixed_n', 'SQR', 'ASQR'], save_strs):
    avg_perf, avg_perf_per_tstep = compute_performances(save_str)
    horizons = [5, 10]
    query_costs = [1,7,10]
    representative_perfs = np.empty((3, 2, avg_perf.shape[2]))
    for ii, qq in enumerate(query_costs):
        for jj, hh in enumerate(horizons):
            representative_perfs[ii, jj] = avg_perf_per_tstep[qq,hh]
    horizons = [str(hh) for hh in horizons]
    query_costs = [str(qq) for qq in query_costs]
    plot_performance(representative_perfs, label=label, title_extra='chain5', horizons=horizons, query_costs=query_costs)
    legend(loc=4)



# compare performance of SQR/ASQR for different numbers of samples
def bootstrap(dataset, estimator, num_samples=10000, sample_size=None):
    if sample_size is None:
        sample_size = len(dataset)
    bootstrap_inds = np.random.randint(0, len(dataset), sample_size * num_samples)
    bootstrap_exs = [dataset[ind] for ind in bootstrap_inds]
    #import ipdb; ipdb.set_trace()
    return [estimator(bootstrap_exs[samp * sample_size: (samp + 1) * sample_size]) for samp in range(num_samples)]

def best_n_estimator(performances):
    """ 
    performances: shape = (num_trials, num_competitors) represents the performance of each competitor on each trial
    Returns: competitor with best average performance
    """
    return np.argmax(np.mean(np.array(performances), axis=0))


this_avg_perf = avg_perf[10,7]
max_avg_perf = np.max(avg_perf[10,7])


sample_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

SQR_performances = np.array([compute_performances(save_strs[1], which_slice=range(k, k+1))[0] for k in range(3000)])
SQR_performances = np.array([perf[10,7] for perf in SQR_performances])
expected_SQR_performance = [ np.mean([this_avg_perf[best_n] for best_n in  bootstrap(SQR_performances, best_n_estimator, 1000, sample_size)])
                                for sample_size in sample_sizes ]
ASQR_performances = np.array([compute_performances(save_strs[2], which_slice=range(k, k+1))[0] for k in range(3000)])
ASQR_performances = np.array([perf[10,7] for perf in ASQR_performances])
expected_ASQR_performance = [ np.mean([this_avg_perf[best_n] for best_n in  bootstrap(SQR_performances, best_n_estimator, 1000, sample_size)])
                                for sample_size in sample_sizes ]

figure()
title('Average performance of different algorithms for chosing n')
plot(sample_sizes, [max_avg_perf,]*len(sample_sizes), label='best n')
plot(sample_sizes, expected_SQR_performance, label='n chosen via SQR')
plot(sample_sizes, expected_ASQR_performance, label='n chosen via ASQR')
xlabel('number of sampled environments')
ylabel('average performance')
legend(loc=4)
assert False






# grid4
# FIXME: fixed_n performance stuck close to 0?!?!? (do we just need to run more episodes??)
save_strs = [
        '2016-08-25_18:02:43___agent=PSRLLimitedQuery__algorithm=fixed_n__environment=grid4__log_n_max=10__log_num_episodes=15__normalize_rewards=0__num_R_samples=100__query_cost=1.0',
        '2016-08-25_18:02:41___agent=PSRLLimitedQuery__algorithm=SQR__environment=grid4__log_n_max=10__log_num_episodes=15__normalize_rewards=0__num_R_samples=100__query_cost=1.0',
        '2016-08-28_01:41:38___agent=PSRLLimitedQuery__algorithm=ASQR__environment=grid4__log_n_max=10__log_num_episodes=15__num_R_samples=100'
]
save_strs = ['results/results__dk_exp_script.py/' + ss for ss in save_strs]

figure()
for label, save_str in zip(['fixed_n', 'SQR', 'ASQR'], save_strs):
    avg_perf, avg_perf_per_tstep = compute_performances(save_str)
    horizons = [10, 15]
    query_costs = [1,7,10]
    representative_perfs = np.empty((3, 2, avg_perf.shape[2]))
    for ii, qq in enumerate(query_costs):
        for jj, hh in enumerate(horizons):
            representative_perfs[ii, jj] = avg_perf_per_tstep[qq,hh]
    horizons = [str(hh) for hh in horizons]
    query_costs = [str(qq) for qq in query_costs]
    plot_performance(representative_perfs, label=label, title_extra='grid4', horizons=horizons, query_costs=query_costs)
    legend(loc=4)






























# ----------------OLD------------------------
# ----------------OLD------------------------
# ----------------OLD------------------------
# ----------------OLD------------------------
# ----------------OLD------------------------
# PLOTTING UTILITIES:
def multiplot(list_of_curves, transpose=True):
    if transpose:
        list_of_curves = list_of_curves.T
    num_plots = len(list_of_curves) 
    colormap = plt.cm.gist_ncar
    gca().set_color_cycle([colormap(i) for i in np.linspace(0.05, 0.95, num_plots)])
    for curve in list_of_curves:
        plot(curve)

def plot_slices(arr3, axis, transpose=False, nrows=2):
    nplots = nrows**2
    for plot_n, ind in enumerate(range(0, arr3.shape[axis], arr3.shape[axis] / nplots)):
        if plot_n < nplots:
            print ind
            nrows = 2
            subplot(nrows, nrows, plot_n+1)
            if transpose:
                multiplot(arr3.take(ind, axis).T)
            else:
                multiplot(arr3.take(ind, axis))
            scale = np.max(arr3.take(ind, axis))
            ylim(-.4*scale, 1.1*scale)


# TODO: label axes
def fixed_n_plots(avg_perf):
    """avg_perf is a 3d array, indexed by cost (c), num_episodes (h), max_num_queries (n)"""

    print "avg_perf.shape=", avg_perf.shape
    print "N.B. all of the of the values are indices in log-scales"
    print "\n blue is low, pink is high"
    print "cost (c), num_episodes (h), max_num_queries (n)"
        

    # even more plots
    if 1:
        figure()
        suptitle('performance curves for all values of c; x=h, y=perf, curves: n')
        plot_slices(avg_perf, 0)
        figure()
        suptitle('performance curves for all values of c; x=n, y=perf, curves:h')
        plot_slices(avg_perf, 0, 1)
        figure()
        suptitle('performance curves for all values of h; x=c, y=perf, curves: n')
        plot_slices(avg_perf, 1)
        figure()
        suptitle('performance curves for all values of h; x=n, y=perf, curves:c')
        plot_slices(avg_perf, 1, 1)
        figure()
        suptitle('performance curves for all values of n; x=c, y=perf, curves: h')
        plot_slices(avg_perf, 2)
        figure()
        suptitle('performance curves for all values of n; x=h, y=perf, curves:c')
        plot_slices(avg_perf, 2, 1)

    if 0:
        figure()
        suptitle('performance curves for best values of c=query_cost, h=num_episodes (horizon), n=max_num_queries')
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


#fixed_n_plots(avg_perf)
#fixed_n_plots(avg_perf_per_tstep)









#----------------------------OLDER---------------------------
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
    

