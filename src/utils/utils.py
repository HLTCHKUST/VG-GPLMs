import os
import pickle
import numpy as np
import torch.nn as nn
import torch
import math
import random
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

def save(toBeSaved, filename, mode='wb'):
    '''
    save data to pickle file
    '''
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4) # protocol 4 allows large size object, it's the default since python 3.8
    file.close()

def load(filename, mode='rb'):
    '''
    load pickle file
    '''
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def pad_sents(sents, pad_token=0, max_len=512):
    '''
    pad input to max length
    '''
    sents_padded = []
    lens = get_lens(sents)
    max_len = min(max(lens), max_len)
    sents_padded = []
    new_len = []
    for i, l in enumerate(lens):
        if l > max_len:
            l = max_len
        new_len.append(l)
        sents_padded.append(sents[i][:l] + [pad_token] * (max_len - l))
    return sents_padded, new_len

def get_mask(sents, unmask_idx=1, mask_idx=0, max_len=512):
    '''
    make mask for padded input
    '''
    lens = get_lens(sents)
    max_len = min(max(lens), max_len)
    mask = []
    for l in lens:
        if l > max_len:
            l = max_len
        mask.append([unmask_idx] * l + [mask_idx] * (max_len - l))
    return mask


def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len