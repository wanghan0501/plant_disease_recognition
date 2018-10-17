# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/8/8 14:30.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
  def __init__(self, n_channels, growth_rate):
    super(Bottleneck, self).__init__()
    interChannels = 4 * growth_rate
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(interChannels)
    self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3,
                           padding=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out = torch.cat((x, out), 1)
    return out


class SingleLayer(nn.Module):
  def __init__(self, n_channels, growth_rate):
    super(SingleLayer, self).__init__()
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3,
                           padding=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = torch.cat((x, out), 1)
    return out


class Transition(nn.Module):
  def __init__(self, n_channels, n_out_channels):
    super(Transition, self).__init__()
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1,
                           bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = F.avg_pool2d(out, 2)
    return out


class DenseNet(nn.Module):
  def __init__(self, growth_rate, depth, reduction, n_classes, bottleneck):
    super(DenseNet, self).__init__()

    nDenseBlocks = (depth - 5) // 4
    if bottleneck:
      nDenseBlocks //= 2

    nChannels = 2 * growth_rate
    self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                           bias=False)
    self.dense1 = self._make_dense(nChannels, growth_rate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growth_rate
    nOutChannels = int(math.floor(nChannels * reduction))
    self.trans1 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense2 = self._make_dense(nChannels, growth_rate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growth_rate
    nOutChannels = int(math.floor(nChannels * reduction))
    self.trans2 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense3 = self._make_dense(nChannels, growth_rate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growth_rate
    nOutChannels = int(math.floor(nChannels * reduction))
    self.trans3 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense4 = self._make_dense(nChannels, growth_rate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks * growth_rate

    self.bn1 = nn.BatchNorm2d(nChannels)
    self.fc = nn.Linear(nChannels, n_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck):
    layers = []
    for i in range(int(n_dense_blocks)):
      if bottleneck:
        layers.append(Bottleneck(n_channels, growth_rate))
      else:
        layers.append(SingleLayer(n_channels, growth_rate))
      n_channels += growth_rate
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.trans3(self.dense3(out))
    out = self.dense4(out)
    out = F.avg_pool2d(F.relu(self.bn1(out)), 9, 9)
    out = F.dropout(out, p=0.5, training=self.training)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
