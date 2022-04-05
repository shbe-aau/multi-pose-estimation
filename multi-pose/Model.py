import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Network structure inspired by:
# https://arxiv.org/pdf/1708.05628.pdf (see fig. 2)
class Model(nn.Module):
    def __init__(self, num_views=4, num_objects=1, weight_init_name="",
                 finetune_encoder=False, classify_objects=False):
        super(Model, self).__init__()

        self.classify_objects = classify_objects
        self.weight_init_name = weight_init_name
        self.num_views = num_views
        self.num_objects = num_objects
        self.output_size = self.num_views*(6+1)
        self.output_size *= self.num_objects

        # Place-holder for last FC layer in the encoder
        self.finetune_encoder = finetune_encoder
        if(self.finetune_encoder):
            self.encoder = nn.Linear(32768,128)
        else:
            self.encoder = None

        # Upscale lantent vector
        self.l01 = nn.Linear(128,128)
        self.l02 = nn.Linear(128+128,128)
        self.l03 = nn.Linear(128+128,128)
        self.l04 = nn.Linear(128+128,128)

        self.bn01 = nn.BatchNorm1d(128)
        self.bn02 = nn.BatchNorm1d(128)
        self.bn03 = nn.BatchNorm1d(128)
        self.bn04 = nn.BatchNorm1d(128)

        # Regress pose using the feature vector
        self.l1 = nn.Linear(128+128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,self.output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        # Predict classes
        if(self.classify_objects):
            self.c1 = nn.Linear(128, 128)
            self.c2 = nn.Linear(128, 64)
            self.c3 = nn.Linear(64, self.num_objects)

            self.bnc1 = nn.BatchNorm1d(128)
            self.bnc2 = nn.BatchNorm1d(64)

        # Init weights
        self.apply(self.init_weights)

    # Input: x = lantent code
    # Output: y = pose as quaternion
    def forward(self,model_in):
        if(self.finetune_encoder):
            x0 = self.encoder(model_in)
        else:
            x0 = model_in
        x1 = F.relu(self.bn01(self.l01(x0))) #output = 128
        x2 = F.relu(self.bn02(self.l02(torch.cat([x0, x1], dim=1))))
        x3 = F.relu(self.bn03(self.l03(torch.cat([x1, x2], dim=1))))
        x4 = F.relu(self.bn04(self.l04(torch.cat([x2, x3], dim=1))))

        x = F.relu(self.bn1(self.l1(torch.cat([x3, x4], dim=1))))
        x = F.relu(self.bn2(self.l2(x)))
        y = self.l3(x)

        confs = y[:,:self.num_objects*self.num_views].reshape(-1,self.num_objects,self.num_views)
        confs = F.softmax(confs, dim=2)
        confs = confs.reshape(-1,self.num_objects*self.num_views)

        if(self.classify_objects is False):
            return torch.cat([confs, y[:,self.num_objects*self.num_views:]], dim=1)

        # Predict classes
        c1 = F.relu(self.bnc1(self.c1(x0)))
        c2 = F.relu(self.bnc2(self.c2(c1)))
        classes = F.softmax(self.c3(c2), dim=1)

        return torch.cat([classes, confs, y[:,self.num_objects*self.num_views:]], dim=1)


    def init_weights(self, m):
        if(type(m) == nn.Linear):
            if(self.weight_init_name == "kaiming_uniform_leakyrelu_fanout"): #default in pytorch
                torch.nn.init.kaiming_uniform_(m.weight,
                                               a=math.sqrt(5),
                                               mode='fan_out',
                                               nonlinearity='leaky_relu')
            elif(self.weight_init_name == "kaiming_normal_leakyrelu_fanout"):
                torch.nn.init.kaiming_normal_(m.weight,
                                               a=math.sqrt(5),
                                               mode='fan_out',
                                               nonlinearity='leaky_relu')
            elif(self.weight_init_name == "kaiming_uniform_leakyrelu_fanin"):
                torch.nn.init.kaiming_uniform_(m.weight,
                                               a=math.sqrt(5),
                                               mode='fan_in',
                                               nonlinearity='leaky_relu')
            elif(self.weight_init_name == "kaiming_uniform_relu_fanout"):
                torch.nn.init.kaiming_uniform_(m.weight,
                                               a=math.sqrt(5),
                                               mode='fan_out',
                                               nonlinearity='relu')
            elif(self.weight_init_name == "orthogonal"):
                # See: https://arxiv.org/pdf/1312.6120.pdf
                # and: https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers
                torch.nn.init.orthogonal_(m.weight,
                                          torch.nn.init.calculate_gain("relu"))
            else:
                #print("No weight init function specified. Using pytorch's default one!")
                return
