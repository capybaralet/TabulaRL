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

def make_squaregrid(T, grid_width): 
    rewards = np.zeros((grid_width, grid_width))
    rewards[1,1] = 1
    rewards = rewards.ravel()
    return "squaregrid", gridworld.make_gridworld(grid_width,T, rewards, reward_noise=0), compute_qMax_grid(T, grid_width)

def compute_qMax_grid(T, grid_width): 
    v = np.zeros((T+1, grid_width* grid_width))

    def row_and_column(state, grid_width):
            return state / grid_width, state % grid_width

    for t in range(1, T+1):
        for s in range(grid_width*grid_width):
            row, column = row_and_column(s, grid_width)

            v[t, s] = max(0, 
                    t - np.abs(1-row) - np.abs(1-column))

    return v[::-1,:]
        

def river(T, n):
    return "river", environment.make_riverSwim(T, n), compute_qMax(T, n)

environments = [
        river(T=20, n=6),
        make_squaregrid(T=20, grid_width=6)
        ]

small_environments = [
        river(T=5, n=3),
        make_squaregrid(T=3, grid_width=2)
        ]

def test_qMax(): 
    for envname, env, qMaxActual in environments: 

        query_function = query_functions.QueryFirstNVisits(0, np.inf)
        agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                                  P_true=env.P, R_true=False, query_function=query_function, tau=100**2)

        R = gridworld.R_normal_dist_to_expectation(env.R)
        qV, qMax = agent.compute_qVals(R , env.P)

        for v,vActual in zip(qMax.values(), qMaxActual) :
            print v
            print vActual

        print envname
        np.testing.assert_almost_equal(
                np.array(qMax.values()), 
                qMaxActual,
                decimal=6)

def test_riverSwim_known_r(): 
    for envname, env, qMaxActual in environments: 

        query_function = query_functions.QueryFirstNVisits(0, np.inf)
        agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
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


def test_much_iteration(): 
    nquery = 10
    reward_tau=10000**2
    for envname, env, qMaxActual in small_environments: 

        query_function = query_functions.QueryFirstNVisits(0, nquery)
        agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                                 P_true=env.P, R_true=None, query_function=query_function, tau=reward_tau)

        f_ext = feature_extractor.FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

        experiment.run_finite_tabular_experiment(agent, env, f_ext, 10000, seed, targetPath='file.csv')   

        qMax = agent.qMax

        for v,vActual in zip(qMax.values(), qMaxActual) :
            print v
            print vActual

        print envname, agent.query_function.visit_count

        np.testing.assert_almost_equal(
                np.array(qMax.values()), 
                qMaxActual,
                decimal=2)

def test_policy_change(): 
    nquery = 10
    reward_tau=10000**2
    for envname, env, qMaxActual in small_environments[2:]: 

        query_function = query_functions.QueryFirstNVisits(0, nquery)
        agent = finite_tabular_agents.PSRLLimitedQuery(env.nState, env.nAction, env.epLen,
                                  P_true=env.P, R_true=None, query_function=query_function, tau=reward_tau)
        

        f_ext = feature_extractor.FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

        experiment.run_finite_tabular_experiment(agent, env, f_ext, 10000, seed, targetPath='file.csv')   

        qMax = np.array(agent.qMax.values())

        for v,vActual in zip(qMax, qMaxActual) :
            print v
            print vActual

        print envname + " visit count: ", agent.query_function.visit_count

        np.testing.assert_almost_equal(
                qMax,
                qMaxActual,
                decimal=3)


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
