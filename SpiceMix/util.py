import os, time, pickle, sys, psutil, resource, datetime, h5py, logging
from collections import Iterable

import numpy as np
import torch
import networkx as nx


pid = os.getpid()
psutil_process = psutil.Process(pid)

# PyTorchDType = torch.float
PyTorchDType = torch.double

def print_datetime():
    return datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]\t')

def array2string(x):
    return np.array2string(x, formatter={'all': '{:.2e}'.format})

def parseSuffix(s):
    return '' if s is None or s == '' else '_' + s

def openH5File(filename, mode='a', num_attempts=5, duration=1):
    for i in range(num_attempts):
        try:
            return h5py.File(filename, mode=mode)
        except OSError as e:
            logging.warning(str(e))
            time.sleep(duration)
    return None

def encode4h5(v):
    if isinstance(v, str): return v.encode('utf-8')
    return v

def parseIiter(g, iiter):
    if iiter < 0:
        iiter += max(map(int, g.keys())) + 1
    return iiter

def zipTensors(*tensors):
    return np.concatenate([
        np.array(a).flatten()
        for a in tensors
    ])

def unzipTensors(arr, shapes):
    assert np.all(arr.shape == (np.sum(list(map(np.prod, shapes))),))
    tensors = []
    for shape in shapes:
        size = np.prod(shape)
        tensors.append(arr[:size].reshape(*shape).squeeze())
        arr = arr[size:]
    return tensors

def save_dict_to_hdf5(filename, dic):
    """
    ....
    """
    with h5py.File(filename, 'a') as h5file:
        save_dict_to_hdf5_group(h5file, '/', dic)

def save_dict_to_hdf5_group(h5file, path, dic):
    """
    ....
    """
    permitted_dtypes = (np.ndarray, np.int64, np.float64, list, bool, float, int, str, bytes)
    for key, item in sorted(dic.items()):
        full_path = path + str(key)
        if isinstance(item, permitted_dtypes):
            h5file[full_path] = item
        elif isinstance(item, dict):
            save_dict_to_hdf5_group(h5file, full_path + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def dict_to_list(dictionary):
    output = []
    dictionary_with_integer_keys = {int(k) : v for k, v in dictionary.items()}
    
    for key, item in sorted(dictionary_with_integer_keys.items()):
        if isinstance(item, dict):
            output.append(dict_to_list(item))
        else:
            output.append(item)
    return output

def list_to_dict(array):
    dictionary = {}
    for key, item in enumerate(array):
        if isinstance(item, list):
            dictionary[key] = list_to_dict(item)
        else:
            dictionary[key] = item
    return dictionary

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return load_dict_from_hdf5_group(h5file, '/')

def load_dict_from_hdf5_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in sorted(h5file[path].items()):
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = load_dict_from_hdf5_group(h5file, path + key + '/')
    return ans
