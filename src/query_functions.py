from collections import defaultdict
import numpy as np

#-------------------------------------------------------------------------------

class QueryFunction(object):
    def __init__(self, queryCost):
        self.__dict__.update(locals())

    def setEnvAgent(self, env, agent):
        self.__dict__.update(locals())

class AlwaysQuery(QueryFunction):
    def __call__(self, state, action, episode, timestep):
        return True, self.queryCost


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
        self.visit_count = defaultdict(lambda :0)

    def __call__(self, state, action, episode, timestep):
        self.visit_count[state, action] += 1
        query = self.visit_count[state, action] <= self.n
        return query, query*self.queryCost

    # We can rewrite all query functions to use this subroutine when called
    def will_query(self, state, action):
        return self.visit_count[state, action] <= self.n

# first n times
class QueryFirstN(QueryFunction):
    def __init__(self, queryCost, n):
        self.__dict__.update(locals())
        self.count = 0

    def __call__(self, state, action, episode, timestep):
        self.count += 1
        query = self.count <= self.n
        return query, query*self.queryCost


class RewardProportional(QueryFunction):
    def __init__(self, queryCost, constant):
        self.__dict__.update(locals())

    def __call__(self, state, action, episode, timestep):
        total_expected = sum(self.agent.R_prior[s, a][0] for s in xrange(self.env.nState) for a in xrange(self.env.nAction))

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
