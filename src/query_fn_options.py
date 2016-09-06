import numpy as np

def estimate_perf(query_fn, agent, env, sampled_rewards, neps):
    agent = copy.deepcopy(agent)
    query_fn = copy.deepcopy(query_fn)
    # RUN EXP


