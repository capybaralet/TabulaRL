from collections import defaultdict
import numpy as np

#-------------------------------------------------------------------------------

class QueryFunction(object):
    def __init__(self, environment, agent, queryCost):
        self.__dict__.update(locals())


class AlwaysQuery(QueryFunction):
    def __call__(self, state, action, episode, timestep):
        return True, self.queryCost


# query with time-dependent probability

class DecayQueryProbability(QueryFunction):
    def __init__(self, environment, agent, queryCost, decay):
        self.__dict__.update(locals())

    def __call__(self, state, action, episode, timestep):
        query = self.probability**timestep < np.random.uniform(), 
        return query, query * self.queryCost


class QueryFirstNVisits(QueryFunction):
    def __init__(self, environment, agent, queryCost, n):
        self.__dict__.update(locals())
        self.visit_count = defaultdict(lambda :0)

    def __call__(self, state, action, episode, timestep):
        self.visit_count[state, action] += 1
        query = self.visit_count[state, action] < self.n
        return query, query*self.queryCost

class QueryFirstN(QueryFunction):
    def __init__(self, environment, agent, queryCost, n):
        self.__dict__.update(locals())
        self.count = 0

    def __call__(self, state, action, episode, timestep):
        self.count += 1
        query = self.count < self.n
        return query, query*self.queryCost
#-------------------------------------------------------------------------------


