# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/7/4 15:36.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.disease_multitask_dataset import DiseaseDataset
from nets.densenet_multitask import densenet201
from utils.log import Logger


class Model:

  def __init__(self, config):
    self.net = densenet201(task1_num_classes=config['multitask_num_classes']['task1'],
                            task2_num_classes=config['multitask_num_classes']['task2'])
    self.config = config
    self.epochs = config['epochs']
    self.use_cuda = config['use_cuda']
    self.alpha = config['alpha']
    if self.use_cuda:
      self.net = self.net.cuda()

    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    self.ckpt_path = os.path.join(config['ckpt_path'], run_timestamp)
    if config['logger']:
      if not os.path.exists(self.ckpt_path):
        os.makedirs(self.ckpt_path)

      self.logger = Logger(os.path.join(self.ckpt_path, 'densenet201_multitask.log')).get_logger()
      self.logger.info(">>>The net is:")
      self.logger.info(self.net)
      self.logger.info(">>>The config is:")
      self.logger.info(json.dumps(self.config, indent=2))
    if config['use_tensorboard']:
      self.run_path = os.path.join(config['run_path'], run_timestamp)
      if not os.path.exists(self.run_path):
        os.makedirs(self.run_path)
        self.writer = SummaryWriter(self.run_path)

  def train(self):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    if self.config['optim'] == 'SGD':
      optimizer = torch.optim.SGD(self.net.parameters(),
                                  lr=self.config['lr'],
                                  momentum=self.config['momentum'],
                                  weight_decay=self.config['weight_decay'])
      lr_decay = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.config['epochs'])
    elif self.config['optim'] == 'Adam':
      optimizer = torch.optim.Adam(self.net.parameters(),
                                   lr=self.config['lr'],
                                   weight_decay=self.config['weight_decay'])
    elif self.config['optim'] == 'Adadelta':
      optimizer = torch.optim.Adadelta(self.net.parameters(),
                                       lr=self.config['lr'],
                                       weight_decay=self.config['weight_decay'])

    train_dataset = DiseaseDataset('train', self.config)
    train_loader = DataLoader(
      train_dataset,
      batch_size=self.config['batch_size'],
      shuffle=True,
      num_workers=self.config['worker'],
      drop_last=True,
      pin_memory=True)

    best_acc = 0
    best_epoch = 0
    for epoch_id in range(self.config['epochs']):
      if self.config['optim'] == 'SGD':
        lr_decay.step()
        if self.config['logger']:
          self.logger.info("Epoch {}'s LR is {}".format(epoch_id, optimizer.param_groups[0]['lr']))

      train_loss = 0
      train_acc = 0
      self.net.train()
      for batch_id, (data, target1, target2) in enumerate(train_loader):
        if self.use_cuda:
          data, target1, target2 = data.cuda(), target1.cuda(), target2.cuda()

        optimizer.zero_grad()
        logits1, logits2 = self.net(data)
        prob1 = F.softmax(logits1, dim=1)
        loss1 = criterion1(logits1, target1)
        prob2 = F.softmax(logits2, dim=1)
        loss2 = criterion2(logits2, target2)
        loss = loss1 + self.alpha * loss2
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        acc = sum(
          np.argmax(prob1.data.cpu().numpy(), 1) ==
          target1.data.cpu().numpy()) / float(train_loader.batch_size)
        train_acc += acc
      train_acc = train_acc / len(train_loader)
      train_loss = train_loss / len(train_loader)
      if self.config['logger']:
        self.logger.info("[Train] Epoch:{}, Loss:{:.6f}, Accuracy:{:.6f}%".format(
          epoch_id, train_loss, 100. * train_acc))
      if self.config['use_tensorboard']:
        self.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch_id)
        self.writer.add_scalar('train/loss', train_loss, epoch_id)
        self.writer.add_scalar('train/accuracy', 100. * train_acc, epoch_id)

      valid_acc = self.validate(epoch_id)

      # save best model
      if valid_acc > best_acc:
        best_acc = valid_acc
        best_epoch = epoch_id
      self.save(epoch_id)
      self.logger.info('[Info] The max validate accuracy is {:.6f} at epoch {}'.format(
        best_acc,
        best_epoch))

  def validate(self, epoch_id):
    valid_dataset = DiseaseDataset('validate', self.config)
    valid_loader = DataLoader(
      valid_dataset,
      batch_size=self.config['batch_size'],
      shuffle=False,
      num_workers=self.config['worker'],
      pin_memory=True,
      drop_last=True)

    criterion = nn.CrossEntropyLoss()

    valid_loss = 0
    valid_acc = 0
    self.net.eval()
    with torch.no_grad():
      for batch_id, (data, target, _) in enumerate(valid_loader):
        if self.use_cuda:
          data, target = data.cuda(), target.cuda()
        logits, _ = self.net(data)
        prob = F.softmax(logits, dim=1)
        loss = criterion(logits, target)
        valid_loss += loss.data.item()
        acc = sum(
          np.argmax(prob.data.cpu().numpy(), 1) ==
          target.data.cpu().numpy()) / float(valid_loader.batch_size)
        valid_acc += acc

      valid_acc = valid_acc / len(valid_loader)
      valid_loss = valid_loss / len(valid_loader)

      if self.config['logger']:
        self.logger.info("[Validate] Epoch:{}, Loss:{:.6f}, Accuracy:{:.6f}%".format(
          epoch_id, valid_loss, 100. * valid_acc))
      if self.config['use_tensorboard']:
        self.writer.add_scalar('validate/loss', valid_loss, epoch_id)
        self.writer.add_scalar('validate/accuracy', 100. * valid_acc, epoch_id)
    return valid_acc

  def test(self, data):
    self.net.eval()
    with torch.no_grad():
      if self.use_cuda:
        data = data.cuda()
      logits = self.net(data)
      prob = F.softmax(logits, dim=1)
    return prob

  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)
    self.net.load_state_dict(ckpt)
    print(">>> Load model completion.")

  def save(self, epoch):
    torch.save(self.net.state_dict(), os.path.join(self.ckpt_path, '{}.pth'.format(epoch)))
