import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

from src.config import config
from src.anchor_generator import AnchorGenerator

class Train_SSH(nn.Cell):

    def __init__(self, config):
        super().__init__()
        self.dtype = np.float32

        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_base_sizes = config.anchor_base_sizes

        # anchor generator
        self.anchor_generators = []
        for i in range(len(self.anchor_base_sizes)):
            self.anchor_generators.append(
                AnchorGenerator(self.anchor_base_sizes[i], self.anchor_scales[i], self.anchor_ratios)
            )
        self.featmap_sizes = [(config.img_height//8, config.img_width//8),
                              (config.img_height//16, config.img_width//16),
                              (config.img_height//32, config.img_width//32)]
        self.anchor_list = self.get_anchors(self.featmap_sizes)
        pass


    def get_anchors(self, featmap_sizes):
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_base_sizes[i])
            multi_level_anchors += (Tensor(anchors.astype(self.dtype)),)

        return multi_level_anchors


Train_SSH(config)