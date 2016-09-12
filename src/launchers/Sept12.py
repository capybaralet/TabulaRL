import os
import itertools
import numpy as np

"""
TOD0:
    1. Manage logging in here? (e.g. construct save_path and pass it as an arg to the script)
    2. touch a file to say which machine I used
"""
jobs_log = 'Sept12_launched'
os.system('touch ' + jobs_log)
def log_it(job_str, fn=jobs_log):
    with open(fn, 'w') as f_:
        f_.write(job_str + '\n')

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
        " --query_fn_selector=fixed_firstNvisits"])
args_.append([" --enviro=grid4",
        " --enviro=dig_grid4",
        " --enviro=det_chain10",
        " --enviro=longY10"])
args_.append([" --query_cost=10.",
        " --query_cost=1."])
        #" --query_cost=.1"])
args_.append([" --log_num_episodes=" + str(n) for n in np.arange(3,14,2)])

cmd_line_args = ["".join(ss) for ss in itertools.product(*args_)]
jobs = [" python" + exp_script + cmd + " --num_exps=50 --save=1 &" for cmd in cmd_line_args]

machine_ind = 0
for job in jobs[:1]:
    job_str = "ssh " + mila_gpus[machine_ind] + job
    print job_str
    log_it(job_str)
    os.system(job_str)
    machine_ind = (1 + machine_ind) % len(mila_gpus)


