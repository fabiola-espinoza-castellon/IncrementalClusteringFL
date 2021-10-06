import os
import json
import _pickle as cPickle
import numpy as np
import torch


def save_dict(dic, filename, filename_dest):
    with open(os.path.join(filename_dest, filename), 'w') as data:
        json.dump(dic, data)


def read_dict(filename):
    with open(filename, 'r') as data:
        return json.load(data)


def save_arrays(array, filename, filename_dest):
    np.save(os.path.join(filename_dest, filename), np.asarray(array))


def save_users_accs(users, filename, filename_dest):
    users_accs = []
    for u in users:
        users_accs.append(users[u].round_accuracy)
    users_accs = np.vstack((users_accs))
    save_arrays(np.asarray(users_accs), filename, filename_dest)


def save_users_last_round(users, filename, filename_dest):
    users_rounds = []
    for u in users:
        users_rounds.append(users[u].last_round)
    users_accs = np.vstack((users_rounds))
    save_arrays(np.asarray(users_accs), filename, filename_dest)


def save_torch_model(model, filename, filename_dest):
    torch.save(model, os.path.join(filename_dest, filename))


def save_cache(class_):
    return cPickle.loads(cPickle.dumps(class_))


def save_cache_local_mem(class_, filename, filename_dest):
    assert os.path.isdir(os.path.join(filename_dest, 'cache')), 'Cache folder not created'
    with open(os.path.join(filename_dest, 'cache', filename), 'wb') as data:
        cPickle.dump(class_, data)


def load_cache_local_mem(filename, filename_dest):
    assert os.path.isfile(os.path.join(filename_dest, 'cache', filename)), 'File does not exist'
    with open(os.path.join(filename_dest, 'cache', filename), 'rb') as data:
        return cPickle.load(data)


def dump_and_load_local_mem(class_, filename, filename_dest):
    save_cache_local_mem(class_, filename, filename_dest)
    return load_cache_local_mem(filename, filename_dest)