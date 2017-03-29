
import numpy as np

def sample(pvals):
    return np.argmax(np.random.multinomial(1, pvals))

#policy

