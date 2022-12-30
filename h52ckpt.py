import numpy as np
import pandas as pd

import deepdish as dd
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

pretrained_dict = dd.io.load('./last.h5')
new_pre_dict = {}
for k, v in pretrained_dict.items():
    new_k = k.replace("_and_", ".")
    if new_k.endswith("Adam") or new_k.endswith("Adam_1") or new_k.endswith("Momentum"):
        continue
    if new_k.endswith("beta1_power") or new_k.endswith("beta2_power"):
        continue
    if new_k.endswith("global_step"):
        continue
    if type(v) is np.ndarray:
        new_pre_dict[new_k] = v


def read_map():
    data = pd.read_csv('./map.csv')
    data_list = data.values.tolist()
    return data_list


data_list = read_map()
new_params_list = []
for name in data_list:
    new_name, odd_name = name[0], name[1]
    param_dict = {}
    param_dict['name'] = new_name
    param_dict['data'] = Tensor(new_pre_dict[odd_name])
    new_params_list.append(param_dict)
save_checkpoint(new_params_list, 'ssh.ckpt')
