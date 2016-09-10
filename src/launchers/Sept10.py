import os
import itertools

exp_script = ' $HOME/TabulaRL/src/dk_exp_in_the_loop.py'

mila_gpus = []
mila_gpus += ['eos' + str(i) for i in range(1, 8)]
mila_gpus += ['eos' + str(i) for i in range(11, 20)]
mila_gpus += ['leto0' + str(i) for i in range(1, 9)]
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
        " --query_fn_selector=fixed_first25per"])
args_.append([" --enviro=grid4",
        " --enviro=multi_chain4",
        " --enviro=det_chain10",
        " --enviro=longY10"])
args_.append([" --query_cost=10.",
        " --query_cost=1.",
        " --query_cost=.1"])
args_.append([" --log_num_episodes=" + str(n) for n in 8**np.arange(1,5)])

cmd_line_args = ["".join(ss) for ss in itertools.product(args_)]
jobs = [exp_script + cmd + " --num_exps=100 --save=1 &" for cmd in cmd_line_args]

machine_ind = 0
for job in jobs:
    print "ssh " + mila_gpus[machine_ind] + job
    os.system("ssh " + mila_gpus[machine_ind] + job)
    machine_ind += 1



launch_strs = [
'ssh eos11 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=grid4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh eos12 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=multi_chain4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh eos13 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=longY10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh eos14 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=det_chain10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh eos15 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=grid4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh eos16 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=multi_chain4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh eos17 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=ASQR  --enviro=longY10 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto01 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=grid4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto02 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=multi_chain4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto03 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=longY10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto04 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=det_chain10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto05 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=grid4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto06 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=multi_chain4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto07 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=longY10 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto08 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_greedy  --enviro=det_chain10 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto11 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=grid4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto12 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=multi_chain4 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto13 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=longY10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto14 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=det_chain10 --num_exps=100 --query_cost=1. --save=1 ',
'ssh leto51 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=grid4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto16 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=multi_chain4 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto17 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=longY10 --num_exps=100 --query_cost=10. --save=1 ',
'ssh leto50 python $HOME/TabulaRL/src/dk_exp_in_the_loop.py --query_fn_selector=OPSRL_omni  --enviro=det_chain10 --num_exps=100 --query_cost=10. --save=1 ']


