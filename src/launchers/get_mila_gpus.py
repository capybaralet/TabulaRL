import os

mila_gpus = []
mila_gpus += ['eos' + str(i) for i in range(1, 8)]
mila_gpus += ['eos' + str(i) for i in range(11, 20)]
mila_gpus += ['leto0' + str(i) for i in range(1, 9)]
#mila_gpus += ['leto0' + str(i) for i in range(1, 7)]
mila_gpus += ['leto' + str(i) for i in range(11, 17)]
mila_gpus += ['leto5' + str(i) for i in range(3)]
mila_gpus += ['bart' + str(i) for i in range(1, 8)]

unreachable = []
for machine in mila_gpus:
    try: #FIXME: this command works even if ssh doesn't!
        os.system('ssh ' + machine + ' echo ' + machine + ' &')
    except:
        unreachable.append(machine)

print unreachable
