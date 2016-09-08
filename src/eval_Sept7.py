from pylab import *
import os
import cPickle as pickle

def pyload(filepath):
    with open(filepath, 'rb') as file_:
        return pickle.load(file_)

save_dir = '/Users/david/TabulaRL/src/results/results__dk_exp_in_the_loop.py/'
dirs = [dd for dd in os.listdir(save_dir)]

ns = 2**np.arange(11)

def state_visits(visit_count, nState):
    return np.array([sum([visit_count[key] for key in visit_count if key[0] == nn]) for nn in range(nState)])

for envv in ['longY10', 'chain10', 'multi_chain4', 'grid4']:
    for query_cost in [10., .1]:
        figure()
        suptitle('env=' + envv + '  query_cost=' + str(query_cost))
        for alg in ['OPSRL_greedy', 'OPSRL_omni', 'ASQR']:
            if 1:#try:
                path = save_dir + [dd for dd in os.listdir(save_dir) if alg in dd and envv in dd and str(query_cost) in dd][0]
                ret = np.load(path + '/returns.npy')
                nq = np.load(path + '/num_queries.npy')
                avg_perf = ret - query_cost * nq
                subplot(121)
                plot(np.zeros(len(ns)), 'k')
                plot(avg_perf.mean(0) / ns, label=alg)
                xticks(range(len(ns)), ns)
                #ylim(-10,3)
                legend(loc=4)
                subplot(122)
                xticks(range(len(ns)), ns)
                ylim(-1,5)
                plot(nq.mean(0) / ns, label=alg)
                # state_visits
            else:#except:
                assert False
                print alg, envv, query_cost, "failed to load"


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



