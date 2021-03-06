from pylab import *
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
import os
import cPickle as pickle

def pyload(filepath):
    with open(filepath, 'rb') as file_:
        return pickle.load(file_)


# TODO: make a plot function that programmatically:
#   maximizes windows, 
#       figure(figsize=(10,9))
#   increases line widths, 
#       plot(range(3), linewidth=2)
#   increases font sizes
#       suptitle('environment=' + envv + "    query_cost= 1 (left), 10 (right)", fontsize=16)
#   adjusts spacing
#       subplots_adjust(left=.05, bottom=.05, right=.99, top=.91, wspace=.11)


# MODIFY HERE FOR DIFFERENT BATCHES OF EXPS!
save_dir = '/home/davidkrueger/results_Sept12/'
#'/Users/david/TabulaRL/src/results/results__dk_exp_in_the_loop.py/Sept9/'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str, default=None)
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

dirs = [dd for dd in os.listdir(save_dir)]

envs = ['longY10', 'det_chain10', 'grid4'] #'multi_chain4',
algs = ['VOI_PSRL_greedy', 'VOI_PSRL_omni', 'ASQR', 'fixed_ASQR']#, 'fixed_first25visits']#, 'fixed_first25', 'fixed_always']
#algs = ['VOI_PSRL_greedy', 'VOI_PSRL_omni', 'ASQR', 'fixed_ASQR', 'fixed_firstNvisits']#, 'fixed_first25', 'fixed_always']
algs_lookup = {'VOI_PSRL_greedy': 'Greedy VoI',
               'VOI_PSRL_omni': 'Omniscient VoI',
               'ASQR': 'dynamic ASQR',
               'fixed_firstNvisits': 'fixed_firstNvisits',
               'fixed_ASQR': 'fixed ASQR'}
#algs = ['ASQR', 'fixed_ASQR']
#algs = ['VOI_PSRL_greedy', 'VOI_PSRL_omni']
query_costs = [1., 10.]

cutoff = 40
h_max = 14
log_num_eps = np.arange(3,h_max,2)
all_ns = 2**np.arange(h_max)
ns = 2**log_num_eps

# plot with errorbars
def my_plot(arr, **kwargs):
    means = arr.mean(axis=1)
    stds = arr.std(axis=1) / arr.shape[1]**.5
    errorbar(range(len(arr)), means, stds, **kwargs) 


# plot performance as a function of num_episodes
if 1:
        for envv in envs:
            figure(figsize=(10,9))
            suptitle(' Performance for different horizons:  environment=' + envv + "  query_cost= 1 (left), 10 (right)", fontsize=14)
            for alg in algs:
                for column, query_cost in enumerate(query_costs):
                    rets = []
                    nqs = []
                    avg_perfs = []
                    perfs = []
                    for log_num_episodes in log_num_eps:
                        try:
                            path = save_dir + [dd for dd in os.listdir(save_dir) if 
                                                        alg in dd and 
                                                        not "fixed_" + alg in dd and 
                                                        envv in dd and 
                                                        str(query_cost) in dd and 
                                                        'log_num_episodes=' + str(log_num_episodes) in dd][0]
                            #assert os.path.exists(path + '/FINISHED')
                            
                            ret = np.load(path + '/returns.npy')[:cutoff,-1]
                            nq = np.load(path + '/num_queries.npy')[:cutoff,-1]
                            rets.append(ret)
                            nqs.append(nq)
                            avg_perfs.append(rets[-1].mean() - query_cost * nqs[-1].mean())
                            perfs.append(rets[-1] - query_cost * nqs[-1])
                            #print "NUM_RETS", alg, envv, query_cost, log_num_episodes
                            #print np.sum(ret == 0)
                        except:
                            print "failed to load", alg, envv, query_cost, log_num_episodes
                            rets.append(10 * 2**log_num_episodes * np.ones(rets[-1].shape))
                            nqs.append(10*2**log_num_episodes * np.ones(nqs[-1].shape))
                            perfs.append(10*2**log_num_episodes * np.ones(perfs[-1].shape))
                            #rets.append(-1*np.ones(rets[-1].shape))
                        #max(nqs[-1].mean())

                    # MAKE PLOTS:
                    subplot(3,2,column+1)
                    title('avg performance per episode')
                    xticks(range(len(ns)), ns)
                    xlim(-.5, len(ns) - .5)
                    plot(np.zeros(len(ns)), 'k-', label="Don't Query")
                    #my_plot(avg_perfs / ns, label=alg)
                    my_plot(perfs / ns.reshape((-1,1)), label=algs_lookup[alg])
                    ###
                    subplot(3,2,column+3)
                    title('avg # of queries per episode')
                    xticks(range(len(ns)), ns)
                    xlim(-.5, len(ns) - .5)
                    if not column:
                        nq_ylim = gca().get_ylim()
                    else:
                        ylim(nq_ylim)
                    my_plot(nqs / ns.reshape((-1,1)), label=algs_lookup[alg])
                    ###
                    subplot(3,2,column+5)
                    title('avg returns per episode')
                    xlabel('horizon (number of episodes)')
                    xticks(range(len(ns)), ns)
                    xlim(-.5, len(ns) - .5)
                    if envv == 'grid4':
                        ylim(-2, 2)
                    else:
                        ylim(-1, 1)
                    my_plot(rets / ns.reshape((-1,1)), label=algs_lookup[alg])
                    plot(np.zeros(len(ns)), 'k-')
            plot(np.zeros(len(ns)), 'k-', label="Don't Query")
            legend(loc=4, prop={'size':8})
            subplots_adjust(left=.05, bottom=.07, right=.99, top=.91, wspace=.11, hspace=.25)
            savefig(envv + "_Sept13")


# --------------------------------------------------------------------
# plot performance as a function of num_episodes FOR A SINGLE HORIZON!
# TODO: title, etc.
# TODO: seem to be missing some of the results here??
#   same results for ASQR itl and ASQR??

def max_R(envv):
    if envv == 'grid4':
        return 2
    elif envv in ['longY10', 'det_chain10']:
        return 1

rm_up_to = 4
ns = all_ns
log_num_episodes = len(all_ns) - 1
if 1:
        for envv in envs:
            figure(figsize=(10,9))
            suptitle('Regret throughout learning:  environment=' + envv + "  query_cost= 1 (left), 10 (right)", fontsize=16)
            for alg in algs:
                for column, query_cost in enumerate(query_costs):
                    rets = []
                    nqs = []
                    perfs = []
                    path = save_dir + [dd for dd in os.listdir(save_dir) if 
                                                alg in dd and 
                                                not "fixed_" + alg in dd and 
                                                envv in dd and 
                                                str(query_cost) in dd and 
                                                'log_num_episodes=' + str(log_num_episodes) in dd][0]
                    # everything is PER EPISODE!
                    # shape = (nexps, nsteps)
                    if rm_up_to:
                        ns = all_ns
                    rets = np.load(path + '/returns.npy')[:cutoff] / ns.reshape((1, -1))
                    nqs = np.load(path + '/num_queries.npy')[:cutoff] / ns.reshape((1, -1))
                    perfs = rets - query_cost * nqs
                    perfs = max_R(envv) - perfs
                    if rm_up_to:
                        rets = rets[:, rm_up_to:]
                        nqs = nqs[:, rm_up_to:]
                        perfs = perfs[:, rm_up_to:]
                        ns = ns[rm_up_to:]


                    # MAKE PLOTS:
                    subplot(3,2,column+1)
                    title('avg regret per episode')
                    xticks(range(len(ns))[1::2], ns[1::2])
                    xlim(-.5, len(ns) - .5)
                    plot(max_R(envv) * np.ones(len(ns)), 'k-', label="Don't Query")
                    means = perfs.mean(0)
                    stds = perfs.mean(0) / len(perfs)**.5
                    errorbar(range(len(ns)), means, stds, label=algs_lookup[alg])
                    ylim(-.2 * gca().get_ylim()[1], min(gca().get_ylim()[1], 10))
                    ###
                    subplot(3,2,column+3)
                    title('avg # of queries per episode')
                    xticks(range(len(ns))[1::2], ns[1::2])
                    xlim(-.5, len(ns) - .5)
                    means = nqs.mean(0)
                    stds = nqs.mean(0) / len(nqs)**.5
                    errorbar(range(len(ns)), means, stds, label=algs_lookup[alg])
                    if not column:
                        nq_ylim = gca().get_ylim()
                    else:
                        ylim(nq_ylim)
                    ###
                    subplot(3,2,column+5)
                    title('avg returns per episode')
                    xlabel('current episode')
                    xticks(range(len(ns))[1::2], ns[1::2])
                    xlim(-.5, len(ns) - .5)
                    if envv == 'grid4':
                        ylim(-2, 2)
                    else:
                        ylim(-1, 1)
                    means = rets.mean(0)
                    stds = rets.mean(0) / len(rets)**.5
                    errorbar(range(len(ns)), means, stds, label=algs_lookup[alg])
                    plot(np.zeros(len(ns)), 'k-')

                    if 0:#alg == 'ASQR':
                        import ipdb; ipdb.set_trace()

            plot(np.zeros(len(ns)), 'k-', label="Don't Query")
            legend(loc=4, prop={'size':8})
            subplots_adjust(left=.05, bottom=.07, right=.99, top=.91, wspace=.11, hspace=.25)
            savefig(envv + "_regret_Sept13")


def average_state_visits(exp_log):
    #return np.mean(np.array([ [exp_log[kk][tstep]['state_visits'] for tstep in sorted(exp_log[kk].keys())] for kk in exp_log.keys()])[0:1], axis=0)
    return np.mean(np.array([ [exp_log[kk][tstep]['state_visits'] for tstep in sorted(exp_log[kk].keys())] for kk in exp_log.keys()]), axis=0)

# plot visit counts throughout learning
if 0:
    for query_cost in query_costs:
        for envv in envs:
            figure()
            suptitle('env=' + envv + '  query_cost=' + str(query_cost))
            for alg in algs:
                for log_num_episodes in [log_num_eps[-1]]:
                    path = save_dir + [dd for dd in os.listdir(save_dir) if 
                                                alg in dd and 
                                                envv in dd and 
                                                str(query_cost) in dd and 
                                                'log_num_episodes=' + str(log_num_episodes) in dd][0]
                    figure()
                    suptitle('env=' + envv + '  query_cost=' + str(query_cost) + '  alg=' + alg)
                    exp_log = pyload(path + '/exp_log')
                    if '4' in envv:
                        svs = [np.zeros((5,4))]
                        svs.extend([np.hstack((average_state_visits(exp_log), np.zeros((13,3)))).reshape((13,5,4))[nn] for nn in range(max(log_num_eps))])
                        for ii in range(max(log_num_eps)):
                            subplot(5,3,ii+1)
                            imshow(svs[ii+1] - svs[ii], cmap="Greys", interpolation='none')
                    else:
                        svs = [np.zeros((1,10))]
                        svs.extend([(average_state_visits(exp_log)[nn]).reshape((1,-1)) for nn in range(max(log_num_eps))])
                        for ii in range(max(log_num_eps)):
                            subplot(13,1,ii+1)
                            imshow(svs[ii+1] - svs[ii], cmap="Greys", interpolation='none')
                            #imshow(svs[ii], cmap="Greys", interpolation='none')
                        #import ipdb; ipdb.set_trace()





