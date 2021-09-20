import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):


    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file, allow_pickle=True)

        self.obj1_18_v2_conv2d_Conv2D = self.__conv(2, name='obj1_18_v2/conv2d/Conv2D', in_channels=3, out_channels=128, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.obj1_18_v2_batch_normalization_FusedBatchNorm = self.__batch_normalization(2, 'obj1_18_v2/batch_normalization/FusedBatchNorm', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.obj1_18_v2_conv2d_1_Conv2D = self.__conv(2, name='obj1_18_v2/conv2d_1/Conv2D', in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.obj1_18_v2_batch_normalization_1_FusedBatchNorm = self.__batch_normalization(2, 'obj1_18_v2/batch_normalization_1/FusedBatchNorm', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.obj1_18_v2_conv2d_2_Conv2D = self.__conv(2, name='obj1_18_v2/conv2d_2/Conv2D', in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.obj1_18_v2_batch_normalization_2_FusedBatchNorm = self.__batch_normalization(2, 'obj1_18_v2/batch_normalization_2/FusedBatchNorm', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.obj1_18_v2_conv2d_3_Conv2D = self.__conv(2, name='obj1_18_v2/conv2d_3/Conv2D', in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.obj1_18_v2_batch_normalization_3_FusedBatchNorm = self.__batch_normalization(2, 'obj1_18_v2/batch_normalization_3/FusedBatchNorm', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.obj1_18_v2_dense_MatMul = self.__dense(name = 'obj1_18_v2/dense/MatMul', in_features = 32768, out_features = 128, bias = True)

    def forward(self, x):
        obj1_18_v2_Flatten_flatten_Reshape_shape_1 = torch.tensor(-1, dtype=torch.int32)
        obj1_18_v2_conv2d_Conv2D_pad = F.pad(x, (1, 2, 1, 2))
        obj1_18_v2_conv2d_Conv2D = self.obj1_18_v2_conv2d_Conv2D(obj1_18_v2_conv2d_Conv2D_pad)
        obj1_18_v2_conv2d_Relu = F.relu(obj1_18_v2_conv2d_Conv2D)
        obj1_18_v2_batch_normalization_FusedBatchNorm = self.obj1_18_v2_batch_normalization_FusedBatchNorm(obj1_18_v2_conv2d_Relu)
        obj1_18_v2_conv2d_1_Conv2D_pad = F.pad(obj1_18_v2_batch_normalization_FusedBatchNorm, (1, 2, 1, 2))
        obj1_18_v2_conv2d_1_Conv2D = self.obj1_18_v2_conv2d_1_Conv2D(obj1_18_v2_conv2d_1_Conv2D_pad)
        obj1_18_v2_conv2d_1_Relu = F.relu(obj1_18_v2_conv2d_1_Conv2D)
        obj1_18_v2_batch_normalization_1_FusedBatchNorm = self.obj1_18_v2_batch_normalization_1_FusedBatchNorm(obj1_18_v2_conv2d_1_Relu)
        obj1_18_v2_conv2d_2_Conv2D_pad = F.pad(obj1_18_v2_batch_normalization_1_FusedBatchNorm, (1, 2, 1, 2))
        obj1_18_v2_conv2d_2_Conv2D = self.obj1_18_v2_conv2d_2_Conv2D(obj1_18_v2_conv2d_2_Conv2D_pad)
        obj1_18_v2_conv2d_2_Relu = F.relu(obj1_18_v2_conv2d_2_Conv2D)
        obj1_18_v2_batch_normalization_2_FusedBatchNorm = self.obj1_18_v2_batch_normalization_2_FusedBatchNorm(obj1_18_v2_conv2d_2_Relu)
        obj1_18_v2_conv2d_3_Conv2D_pad = F.pad(obj1_18_v2_batch_normalization_2_FusedBatchNorm, (1, 2, 1, 2))
        obj1_18_v2_conv2d_3_Conv2D = self.obj1_18_v2_conv2d_3_Conv2D(obj1_18_v2_conv2d_3_Conv2D_pad)
        obj1_18_v2_conv2d_3_Relu = F.relu(obj1_18_v2_conv2d_3_Conv2D)
        obj1_18_v2_batch_normalization_3_FusedBatchNorm = self.obj1_18_v2_batch_normalization_3_FusedBatchNorm(obj1_18_v2_conv2d_3_Relu)
        obj1_18_v2_Flatten_flatten_Shape = list(obj1_18_v2_batch_normalization_3_FusedBatchNorm.size())
        #obj1_18_v2_Flatten_flatten_Reshape = torch.reshape(input = obj1_18_v2_batch_normalization_3_FusedBatchNorm, shape = (-1,32768))
        intermediate = obj1_18_v2_batch_normalization_3_FusedBatchNorm.permute(0,2,3,1)
        obj1_18_v2_Flatten_flatten_Reshape = torch.reshape(input = intermediate, shape = (-1,32768))

        obj1_18_v2_Flatten_flatten_strided_slice = obj1_18_v2_Flatten_flatten_Shape[0:1]
        obj1_18_v2_dense_MatMul = self.obj1_18_v2_dense_MatMul(obj1_18_v2_Flatten_flatten_Reshape)
        obj1_18_v2_Flatten_flatten_Reshape_shape = [obj1_18_v2_Flatten_flatten_strided_slice,obj1_18_v2_Flatten_flatten_Reshape_shape_1]
        return obj1_18_v2_dense_MatMul


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
