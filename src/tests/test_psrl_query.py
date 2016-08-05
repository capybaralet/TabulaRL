import numpy as np
from TabulaRL import experiment, query_functions, finite_tabular_agents, environment, gridworld, feature_extractor

seed=np.random.randint(10000)

def compute_qMax(T, length):
    v = np.zeros((T+1, length), 'float128')

    
    for t in range(1, T+1):
        for i in range(length):
            if i == length-1:
                v[t, i] = 1 + np.dot([.4, .6], v[t-1, [i-1, i  ]])
            elif i == 0:
                v[t, i] =     np.dot([.4, .6], v[t-1, [i  , i+1]])
            else:
                v[t, i] =     np.dot([.05, .35, .6], v[t-1, [i-1, i, i+1]])
                v[t, i] = (
                        np.float128('.05')   * v[t-1, i-1] +
                        np.float128('.6')    * v[t-1, i  ] +
                        np.float128('.35')   * v[t-1, i+1])

                                        
    return v[::-1,:]


def test_riverSwim_qMax(): 
    T =20
    n =6
    env = environment.make_riverSwim(T, n)
    qMaxActual = compute_qMax(T, n)

    query_function = query_functions.QueryFirstNVisits(0, np.inf)
    agent = finite_tabular_agents.PSRL(env.nState, env.nAction, env.epLen,
                              P_true=env.P, R_true=False, query_function=query_function)

    R = gridworld.R_normal_dist_to_expectation(env.R)
    qV, qMax = agent.compute_qVals(R , env.P)

    for v,vActual in zip(qMax.values(), qMaxActual) :
        print v
        print vActual

    np.testing.assert_almost_equal(
            np.array(qMax.values()), 
            qMaxActual,
            decimal=6)

def test_riverSwim_known_r(): 
    T =20
    n =6
    env = environment.make_riverSwim(T, n)
    qMaxActual = compute_qMax(T, n)

    query_function = query_functions.QueryFirstNVisits(0, np.inf)
    agent = finite_tabular_agents.PSRL(env.nState, env.nAction, env.epLen,
                              P_true=env.P, R_true=env.R, query_function=query_function)

    agent.update_policy()
    qMax = agent.qMax

    for v,vActual in zip(qMax.values(), qMaxActual) :
        print v
        print vActual

    np.testing.assert_almost_equal(
            np.array(qMax.values()), 
            qMaxActual,
            decimal=6)


def test_riverSwim_much_iteration(): 
    T =5
    n =3
    env = environment.make_riverSwim(T, n)
    qMaxActual = compute_qMax(T, n)

    query_function = query_functions.QueryFirstNVisits(0, 1000)
    agent = finite_tabular_agents.PSRL(env.nState, env.nAction, env.epLen,
                              P_true=env.P, R_true=None, query_function=query_function)

    f_ext = feature_extractor.FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    experiment.run_finite_tabular_experiment(agent, env, f_ext, 10000, seed, targetPath='file.csv')   

    qMax = agent.qMax

    for v,vActual in zip(qMax.values(), qMaxActual) :
        print v
        print vActual

    print agent.query_function.visit_count

    np.testing.assert_almost_equal(
            np.array(qMax.values()), 
            qMaxActual,
            decimal=2)

def test_riverSwim_much_iteration(): 
    nquery = 1000
    T =5
    n =3
    env = environment.make_riverSwim(T, n)
    qMaxActual = compute_qMax(T, n)

    query_function = query_functions.QueryFirstNVisits(0, nquery)
    agent = finite_tabular_agents.PSRL(env.nState, env.nAction, env.epLen,
                              P_true=env.P, R_true=None, query_function=query_function)
    

    f_ext = feature_extractor.FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    experiment.run_finite_tabular_experiment(agent, env, f_ext, 10000, seed, targetPath='file.csv')   

    qMax = np.array(agent.qMax.values())

    for v,vActual in zip(qMax, qMaxActual) :
        print v
        print vActual

    print "visit count: ", agent.query_function.visit_count

    np.testing.assert_almost_equal(
            qMax,
            qMaxActual,
            decimal=2)


    #qValues should change since not all visits have maxed out 
    qV = np.array(agent.qVals.values())
    agent.update_policy()
    qV2 = np.array(agent.qVals.values())

    np.any(np.abs(qV - qV2)>1e-8)

    for k in agent.query_function.visit_count.keys():
        agent.query_function.visit_count[k] += nquery

    #now qvalues shouldn't change
    agent.update_policy()
    qV = np.array(agent.qVals.values())
    agent.update_policy()
    qV2 = np.array(agent.qVals.values())

    np.testing.assert_almost_equal(
            qV,
            qV2,
            decimal=8)


