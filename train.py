import os
import cv2
import time
import numpy as np

from mindspore.nn import SGD
from mindspore import context
import mindspore.common.dtype as mstype

from src.config import config
from src.train_ssh import TrainSSH
from src.network_define import LossNet
from src.dataset import data_to_mindrecord_byte_image, create_wider_dataset


rank = 0
device_num = 1
context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=0)

def prepare_wider_dataset():
    """ prepare wider dataset """
    print("Start create dataset!")

    prefix = "wider.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)

        if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
            if not os.path.exists(config.image_dir):
                print("Please make sure config:image_dir is valid.")
                raise ValueError(config.image_dir)
            print("Create Mindrecord. It may take some time.")
            data_to_mindrecord_byte_image(config, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    dataset = create_wider_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def train_ssh():
    dataset_size, dataset = prepare_wider_dataset()

    net = TrainSSH(config)
    net = net.set_train()

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)


train_ssh()
'''
dataset_size, dataset = prepare_wider_dataset()
print(dataset_size)
for t in dataset.create_tuple_iterator():
    img = t[0]
    c = t[2][0][0]
    img = img.asnumpy()[0]
    img = np.transpose(img, (1, 2, 0))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img *= 255
    cv2.imwrite('./tmp.jpg', np.uint8(img))
    tmp = cv2.imread('./tmp.jpg')
    draw_0 = cv2.rectangle(tmp, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 2)
    cv2.imshow('t', draw_0)
    cv2.waitKey(0)
'''