from pylab import *
import os
import cPickle as pickle

def pyload(filepath):
    with open(filepath, 'rb') as file_:
        return pickle.load(file_)

# MODIFY HERE FOR DIFFERENT BATCHES OF EXPS!
save_dir = '/Users/david/TabulaRL/src/results/results__dk_exp_in_the_loop.py/Sept9/'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=float, default=None)
args = parser.parse_args()
args_dict = vars(args)
locals().update(args_dict) # add all args to local namespace

dirs = [dd for dd in os.listdir(save_dir)]

envs = ['longY10', 'chain10', 'multi_chain4', 'grid4']
algs = ['VOI_PSRL_greedy', 'VOI_PSRL_omni', 'ASQR', 'fixed_ASQR', 'fixed_first25', 'fixed_first25visits', 'fixed_always']
query_costs = [10., 1.]
log_num_eps = range(3,13,3)
ns = 2**log_num_eps

# plot performance as a function of num_episodes
for envv in envs:
    for query_cost in query_costs:
        figure()
        suptitle('env=' + envv + '  query_cost=' + str(query_cost))
        for alg in algs:
            rets = []
            nqs = []
            avg_perf = []
            for log_num_episodes in log_num_eps:
                path = save_dir + [dd for dd in os.listdir(save_dir) if 
                                            alg in dd and 
                                            envv in dd and 
                                            str(query_cost) in dd and 
                                            'log_num_episodes=' + str(log_num_episodes) in dd][0]
                ret = np.load(path + '/returns.npy')[:,-1].mean()
                nq = np.load(path + '/num_queries.npy')[:,-1].mean()
                rets.append(ret)
                nqs.append(nq)
                avg_perf.append(ret - query_cost * nq)

            # MAKE PLOTS:
            subplot(131)
            title('performance')
            xticks(range(len(ns)), ns)
            plot(np.zeros(len(ns)), 'k-')
            plot(avg_perf / ns, label=alg)
            subplot(132)
            title('num queries')
            xticks(range(len(ns)), ns)
            plot(nq / ns, label=alg)
            ubplot(133)
            title('returns')
            xticks(range(len(ns)), ns)
            plot(np.zeros(len(ns)), 'k-')
            plot(ret / ns, label=alg)
            legend(loc=4)


# TODO
# look at visit counts throughout learning
if 0:
    # TODO: does it make sense to average visit counts?
    for envv in ['longY10', 'chain10', 'multi_chain4', 'grid4']:
        for query_cost in [10., .1]:
            figure() # TODO: combine different algos (different columns of same figure)
            suptitle('env=' + envv + '  query_cost=' + str(query_cost))
            for alg in ['OPSRL_greedy', 'OPSRL_omni', 'ASQR']:
                    path = save_dir + [dd for dd in os.listdir(save_dir) if alg in dd and envv in dd and str(query_cost) in dd][0]
                    figure()
                    suptitle('env=' + envv + '  query_cost=' + str(query_cost) + '  alg=' + alg)
                    exp_log = pyload(path + '/exp_log')
                    if '4' in envv:
                        svs = [np.zeros((4,4))]
                        svs.extend([state_visits(exp_log[int(nn)]['visit_count'], 16).reshape((4,4)) for nn in ns])
                        for ii in range(len(ns)):
                            subplot(4,3,ii+1)
                            imshow(svs[ii+1] - svs[ii], cmap="Greys", interpolation='none')
                    else:
                        svs = [np.zeros((1,10))]
                        svs.extend([state_visits(exp_log[int(nn)]['visit_count'], 10).reshape((1,-1)) for nn in ns])
                        for ii in range(len(ns)):
                            subplot(11,1,ii+1)
                            imshow(svs[ii+1] - svs[ii], cmap="Greys", interpolation='none')
                            #imshow(svs[ii], cmap="Greys", interpolation='none')



