
from inspect import currentframe

def name_print(name):
    frame = currentframe().f_back
    locs, globs = frame.f_locals, frame.f_globals
    value = locs[name] if name in locs else globs.get(name, "???")
    print name, "=", value
    del frame
    return name + "=" + str(value)

n = 42
name_print("n")

def make_save_str(variables, base_str=''):
    return base_str + '____' + [name_print(var) for var in variables].join('__')
