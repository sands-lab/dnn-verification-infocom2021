import os
from .constants import * 


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_suffix(config):
    T = config["rdm"]["fixation_dur"] + config["rdm"]["task_dur"]
    sux = "{}_neurons_{}_steps_{}_noise_{}".format(0, config["network"]["N_rec"] , T/config["rdm"]["dt"], config["rdm"]["noise_variance"])    
    return sux

def parse_nan(s):
    return None if s == NONE  else s

def parse_bool(s):
    return True if s == TRUE else False

