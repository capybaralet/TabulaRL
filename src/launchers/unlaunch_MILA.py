import os
import itertools
import numpy as np

# TODO: test that machines are online
mila_gpus = []
mila_gpus += ['eos' + str(i) for i in range(1, 8)]
mila_gpus += ['eos' + str(i) for i in range(11, 20)]
#mila_gpus += ['leto0' + str(i) for i in range(1, 9)]
mila_gpus += ['leto0' + str(i) for i in range(1, 7)]
mila_gpus += ['leto' + str(i) for i in range(11, 18)]
mila_gpus += ['leto5' + str(i) for i in range(3)]
mila_gpus += ['bart' + str(i) for i in range(1, 8)]

for machine in mila_gpus:
    os.system('ssh ' + machine + ' pkill -u kruegerd &')
