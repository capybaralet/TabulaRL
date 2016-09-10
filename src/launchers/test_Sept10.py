import os
import itertools
import numpy as np

exp_script = ' $HOME/TabulaRL/src/dk_exp_in_the_loop.py'

mila_gpus = []
mila_gpus += ['eos' + str(i) for i in range(1, 8)]
mila_gpus += ['eos' + str(i) for i in range(11, 20)]
#mila_gpus += ['leto0' + str(i) for i in range(1, 9)]
mila_gpus += ['leto0' + str(i) for i in range(1, 7)]
mila_gpus += ['leto' + str(i) for i in range(11, 18)]
mila_gpus += ['leto5' + str(i) for i in range(3)]
mila_gpus += ['bart' + str(i) for i in range(1, 8)]

args_ = []
args_.append([" --query_fn_selector=ASQR",
        " --query_fn_selector=VOI_PSRL_greedy",
        " --query_fn_selector=VOI_PSRL_omni",
        " --query_fn_selector=fixed_ASQR",
        " --query_fn_selector=fixed_always",
        " --query_fn_selector=fixed_first25visits",
        " --query_fn_selector=fixed_first25"])
args_.append([
        " --enviro=longY10"])
args_.append([
        " --query_cost=1."])
        #" --query_cost=.1"])

cmd_line_args = ["".join(ss) for ss in itertools.product(*args_)]
jobs = [" python" + exp_script + cmd + " --num_exps=100 --save=1 &" for cmd in cmd_line_args]

print jobs
print len(jobs)
#assert False

machine_ind = 0
for job in jobs:
    print "ssh " + mila_gpus[machine_ind] + job
    os.system("ssh " + mila_gpus[machine_ind] + job)
    machine_ind = (1 + machine_ind) % len(mila_gpus)


