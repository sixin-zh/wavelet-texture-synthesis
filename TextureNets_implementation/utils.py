try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import os
import numpy as np
import torch
from torchvision import utils
import matplotlib.pyplot as plt
import seaborn as sns

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
   
import hashlib
def hash_str2int2(s):
    if s is not None:
        s = s.encode('UTF-8')
        return int(hashlib.sha1(s).hexdigest(), 16) % (100)
    return 0

def mkdir(outdir):
    try:
        os.mkdir(outdir)
    except:
        print(outdir, 'already exists')   
