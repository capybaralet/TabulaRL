
import numpy as np

#-------------------------------------------------------------------------------

class QueryFunction(object):
    def __init__(self, environment, agent):
        self.__dict__.update(locals())


class AlwaysQuery(QueryFunction):
    def __call__(self, state, action, episode, timestep):
        return True


#-------------------------------------------------------------------------------


