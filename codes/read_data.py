import numpy as np
import os

# data_file = 'parameters.txt'

def global_parameters(data_file,list_ints,list_floats,list_bools,list_chars):
    # script_dir = os.path.dirname(__file__)
    # filename = os.path.join(script_dir, data_file)
    filename = data_file
    with open(filename,'r') as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        line = line.split('\n')[0]
        if ':' in line:
            new_param = line.split(':')[0]
            if new_param in  list_ints:
                globals()[new_param] = int(line.split(':')[1])
            elif new_param in list_floats:
                globals()[new_param] = float(line.split(':')[1])
            elif new_param in list_bools:
                if line.split(':')[1] == 't':
                    globals()[new_param] = True
                elif line.split(':')[1] == 'f':
                    globals()[new_param] = False
            elif new_param in list_chars:
                globals()[new_param] = line.split(':')[1]

    all_ints = [globals()[quantity_int] for quantity_int in list_ints]
    all_floats = [globals()[quantity_float] for quantity_float in list_floats]
    all_bools = [globals()[quantity_bool] for quantity_bool in list_bools]
    all_chars = [globals()[quantity_bool] for quantity_bool in list_chars]

    return all_ints,all_floats,all_bools,all_chars