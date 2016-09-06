from collections import defaultdict # causing problems for pickling
import itertools
import numpy as np

#-------------------------------------------------------------------------------

class QueryFunction(object):
    def __init__(self, queryCost):
        self.__dict__.update(locals())

    def setAgent(self, agent):
        self.__dict__.update(locals())
        self.agent.query_function = self

class AlwaysQuery(QueryFunction):
    def __call__(self, state, action, episode, timestep):
        return True, self.queryCost

    def will_query(self, state, action):
        return True


# query with time-dependent probability
class DecayQueryProbability(QueryFunction):
    def __init__(self, queryCost, decay):
        self.__dict__.update(locals())

    def __call__(self, state, action, episode, timestep):
        query = self.probability**timestep < np.random.uniform(), 
        return query, query * self.queryCost


class QueryFirstNVisits(QueryFunction):
    def __init__(self, queryCost, n):
        self.__dict__.update(locals())
        #self.visit_count = defaultdict(lambda :0)
        #self.query_count = defaultdict(lambda :0)

    def setAgent(self, agent):
        self.__dict__.update(locals())
        self.agent.query_function = self
        self.visit_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}
        self.query_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}

    def __call__(self, state, action, episode, timestep):
        query = self.will_query(state, action)
        if query:
            self.query_count[state, action] += 1
        self.visit_count[state, action] += 1
        return query, query*self.queryCost

    # We can rewrite all query functions to use this subroutine when called
    def will_query(self, state, action):
        return self.visit_count[state, action] < self.n


class QueryFixedFunction(QueryFunction):
    def __init__(self, queryCost, func):
        self.__dict__.update(locals())
        #self.visit_count = defaultdict(lambda :0)
        #self.query_count = defaultdict(lambda :0)

    def setAgent(self, agent):
        self.__dict__.update(locals())
        self.agent.query_function = self
        self.visit_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}
        self.query_count = {sa: 0 for sa in itertools.product(range(self.agent.nState), range(self.agent.nAction))}

    def __call__(self, state, action, episode, timestep):
        query = self.will_query(state, action)
        if query:
            self.query_count[state, action] += 1
        self.visit_count[state, action] += 1
        return query, query*self.queryCost

    #We can rewrite all query functions to use this subroutine when called
    def will_query(self, state, action):
        return self.visit_count[state, action] < self.func(state, action)

class QueryFirstN(QueryFunction):
    def __init__(self, queryCost, n):
        self.__dict__.update(locals())
        self.count = 0

    def __call__(self, state, action, episode, timestep):
        self.count += 1
        query = self.count < self.n
        return query, query*self.queryCost


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
