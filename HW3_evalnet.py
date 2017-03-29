"""
WIP HW 3

"""

import numpy
np  =  numpy
import os
import sys

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

# my code
from algorithms import iterative_policy_evaluation
from environments import MDP

import argparse
parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--env', type=str, default='random')
parser.add_argument('--size', type=int, default=3)
parser.add_argument('--gamma', type=int, default=.5) #
parser.add_argument('--num_train', type=int, default=10000) #
# neural net
#parser.add_argument('--BN', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--num_units', type=int, default=1000)
parser.add_argument('--num_layers', type=int, default=4)
# script config
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=1337)


args = parser.parse_args()
args_dict = args.__dict__
flags = [flag.lstrip('--') for flag in sys.argv[1:]]
flags = [ff for ff in flags if not ff.startswith('save_dir')]


# SET-UP SAVING
save_dir = args_dict.pop('save_dir')
save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
# TODO: use this to save stuff
args_dict['save_path'] = save_path
if args_dict['save']:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')
    print( save_path)
    #assert False
locals().update(args_dict)


# PSEUDO-RANDOMNESS (optional)
if seed is not None:
    np.random.seed(seed)
    rng = numpy.random.RandomState(seed)
else:
    rng = numpy.random.RandomState(np.random.randint(2**32 - 1))



# ----------------------------------------------
nS = nA = size
def randP():
    return rng.dirichlet(1./nS * np.ones(nS), nS * nA).reshape((nS,nA,nS))

def randR():
    return rng.normal(0,1,(nS,nA))


# ----------------------------------------------
# EXPERIMENT 1: fixed MDP

#----------
# MAKE DATA
P = randP()
R = randR()
num_examples = int(num_train * 1.2)
X = rng.dirichlet(1./nA * np.ones(nA), (num_examples, nS))
Y = []
print "making dataset..."
for nn, x in enumerate(X):
    print nn
    Y.append(iterative_policy_evalutation(x, MDP(P, R, gamma)))
Y = numpy.array(Y)
X = X.reshape((num_examples, -1))
# train/valid/test split
trX, trY = X[:num_train], Y[:num_train]
teX, teY = X[num_train:], Y[num_train:]
vaX, vaY = teX[:len(teX)/2], teY[:len(teX)/2]
teX, teY = teX[len(teX)/2:], teY[len(teX)/2:]
# TODO: save dataset



#----------
# build model
model = Sequential()
model.add(Dense(num_units, input_shape=(nS * nA,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
for layer in range(num_layers-1):
    model.add(Dense(num_units))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
model.add(Dense(nS))

model.summary()

model.compile(loss='mse',
              optimizer=Adam())

history = model.fit(trX, trY,
                    batch_size=batch_size, 
                    nb_epoch=num_epochs,
                    verbose=1, 
                    validation_data=(vaX, vaY))



