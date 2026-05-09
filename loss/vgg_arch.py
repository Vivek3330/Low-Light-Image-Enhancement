import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg as vgg

# 🔥 GLOBAL DEVICE (CPU)
device = torch.device("cpu")


class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map)
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco

        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found")
        return ret


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')


VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'

NAMES = {
    'vgg19': [
        'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
        'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
        'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
        'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
        'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5'
    ]
}


@ARCH_REGISTRY.register()
class VGGFeatureExtractor(nn.Module):

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):

        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type]

        # 🔍 find max layer index
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        # 🔧 load pretrained VGG
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=device)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[:max_idx + 1]

        # 🔧 build modified network
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(2, stride=pooling_stride)
            else:
                modified_net[k] = v

        # 🔥 CPU ONLY
        self.vgg_net = nn.Sequential(modified_net).to(device)

        # 🔧 freeze params
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()

        # 🔥 FIXED (NO CUDA)
        if self.use_input_norm:
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
            )
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
            )

    def forward(self, x):

        if self.range_norm:
            x = (x + 1) / 2

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}

        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output