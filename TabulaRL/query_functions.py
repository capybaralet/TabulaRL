from collections import defaultdict # causing problems for pickling
import itertools
import numpy as np

#-------------------------------------------------------------------------------
"""
TODO: it might make more sense to put all of this in the agent 
"""

# timestep \in [0, epLen]

class QueryFunction(object):
    def __init__(self, queryCost):
        self.__dict__.update(locals())

    def setAgent(self, agent):
        self.__dict__.update(locals())
        self.agent.query_function = self
        self.visit_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}
        if self.agent.reward_depends_on_action:
            self.query_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}
        else:
            self.query_count = {sa: 0 for sa in range(self.agent.nState)}

    def __call__(self, state, action, episode, timestep):
        query = self.will_query(state, action, episode, timestep)
        if self.agent.reward_depends_on_action:
            if query:
                self.query_count[state, action] += 1
        else:
            if query:
                self.query_count[state] += 1
        self.visit_count[state, action] += 1
        return query, query*self.queryCost

    def will_query(self, state, action, episode, timestep):
        print "NOT IMPLEMENTED"
        assert False


class AlwaysQuery(QueryFunction):
    def will_query(self, state, action, episode, timestep):
        return True


class QueryFirstN(QueryFunction):
    def __init__(self, queryCost, n):
        self.__dict__.update(locals())

    def will_query(self, state, action, episode, timestep):
        return sum(self.query_count.values()) < self.n


class QueryFirstNVisits(QueryFunction):
    def __init__(self, queryCost, n):
        self.__dict__.update(locals())

    def will_query(self, state, action, episode, timestep):
        if self.agent.reward_depends_on_action:
            return self.query_count[state, action] < self.n
        else:
            return self.query_count[state] < self.n


class QueryFixedFunction(QueryFunction):
    def __init__(self, queryCost, func):
        self.__dict__.update(locals())

    def will_query(self, state, action, episode, timestep):
        if self.agent.reward_depends_on_action:
            return self.query_count[state, action] < self.func(state, action)
        else:
            return self.query_count[state] < self.func(state, action)


# query with time-dependent (decaying) probability
# TODO: would want to know episode length...
class DecayQueryProbability(QueryFunction):
    def __init__(self, queryCost, func):
        self.__dict__.update(locals())

    def will_query(self, state, action, episode, timestep):
        return self.func(episode, timestep) < np.random.uniform()






# TODO: below
class RewardProportional(QueryFunction):
    def __init__(self, queryCost, constant):
        self.__dict__.update(locals())

    def __call__(self, state, action, episode, timestep):
        total_expected = sum(self.agent.R_prior[s, a][0] for s in xrange(self.agent.nState) for a in xrange(self.agent.nAction))

        if abs(total_expected ) > 0:
            proportion = self.agent.R_prior[state, action][0] / total_expected
        else:
            proportion = .5

        query = np.random.binomial(1, self.constant * proportion)
        return query, query*self.queryCost

class EntropyThreshold(QueryFunction):
    def __init__(self, queryCost, constant):
        self.__dict__.update(locals())

    def __call__(self, state, action, episode, timestep):
        tau = self.agent.R_prior[state, action][1]
        entropy = .5*np.log(2/tau*np.pi * np.e)


        query = entropy > self.constant
        return query, query*self.queryCost
