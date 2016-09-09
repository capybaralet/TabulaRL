from pylab import *
import os
import cPickle as pickle

def pyload(filepath):
    with open(filepath, 'rb') as file_:
        return pickle.load(file_)

# MODIFY HERE FOR DIFFERENT BATCHES OF EXPS!
save_dir = '/Users/david/TabulaRL/src/results/results__dk_exp_in_the_loop.py/Sept9/'
ns = 2**np.arange(13)
query_costs = [10., 1.]
query_costs = [1.]

dirs = [dd for dd in os.listdir(save_dir)]

def state_visits(visit_count, nState):
    return np.array([sum([visit_count[key] for key in visit_count if key[0] == nn]) for nn in range(nState)])

# per state, by ASQR
avg_nqs_1 = []
avg_nqs10 = []
for envv in ['longY10', 'chain10', 'multi_chain4', 'grid4']:
    for query_cost in query_costs:
        figure()
        suptitle('env=' + envv + '  query_cost=' + str(query_cost))
        for alg in ['OPSRL_greedy', 'OPSRL_omni', 'ASQR']:
            if 1:#try:
                path = save_dir + [dd for dd in os.listdir(save_dir) if 
                                            alg in dd and 
                                            envv in dd and 
                                            str(query_cost) in dd and 
                                            'update_freq=10' not in dd][0]
                ret = np.load(path + '/returns.npy')
                nq = np.load(path + '/num_queries.npy')
                if alg == 'ASQR':
                    if query_cost == 10:
                        avg_nqs10.append(nq[:,-1].mean())
                    else:
                        avg_nqs_1.append(nq[:,-1].mean())
                avg_perf = ret - query_cost * nq
                subplot(131)
                title('performance')
                plot(np.zeros(len(ns)), 'k')
                plot(avg_perf.mean(0) / ns, label=alg)
                xticks(range(len(ns)), ns)
                subplot(132)
                title('num queries')
                ylim(-1,5)
                plot(nq.mean(0) / ns, label=alg)
                xticks(range(len(ns)), ns)
                #ylim(-10,3)
                subplot(133)
                title('returns')
                plot(np.zeros(len(ns)), 'k')
                plot(ret.mean(0) / ns, label=alg)
                xticks(range(len(ns)), ns)
                legend(loc=4)
                # state_visits
            else:#except:
                assert False
                print alg, envv, query_cost, "failed to load"

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



