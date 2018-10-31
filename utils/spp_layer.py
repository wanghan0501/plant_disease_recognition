"""
Created by Wang Han on 2018/10/31 14:03.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""

import math
import torch
import torch.nn.functional as F


class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        level = 1

        for i in range(self.num_levels):
            level <<= 1

            kernel_size = (math.ceil(h / level),
                           math.ceil(w / level))
            padding = (math.floor((kernel_size[0] * level - h + 1) / 2),
                       math.floor((kernel_size[1] * level - w + 1) / 2))

            zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0],
                                           padding[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new, w_new = x_new.size()[2:]

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # select pool type
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(
                    x_new, kernel_size=kernel_size, stride=stride).view(
                    num, -1)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(
                    x_new, kernel_size=kernel_size, stride=stride).view(
                    num, -1)
            # flatten and concate
            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten
