# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/7/4 15:56.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import argparse
import re

import torch

from models.resnet50v2_sn_model import Model
from utils.gpu import set_gpu
from utils.parse import parse_yaml

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plant Disease Recognition')
  parser.add_argument('--seed', type=int, default=22,
                      help='random seed for training. default=22')
  parser.add_argument('--gpu', default='0', type=str,
                      help='use gpu device. default: 0')
  parser.add_argument('--model', default='all', type=str,
                      help='select one model to train. default: all')

  args = parser.parse_args()

  set_gpu(args.gpu)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  config = parse_yaml()
  config = config['pepper']
  # model = Model(config)
  # ckpt = torch.load('pretrained/resnet50v2_sn.pth')
  # pretrained_dict = ckpt['state_dict']
  # model_dict = model.net.state_dict()
  # patten = re.compile(r'(?!fc)')
  # for key in list(pretrained_dict.keys()):
  #   cur_key = key[7:]
  #   res = patten.match(cur_key)
  #   if res:
  #     model_dict[cur_key] = pretrained_dict[key]
  # model_dict = model.net.load_state_dict(model_dict, strict=False)
  # model.train()
  model = Model(config)
  ckpt = torch.load('multi_task_ckpt/species/2018Oct16-183710/33.pth')
  model_dict = model.net.state_dict()
  patten = re.compile(r'(?!fc)')
  for key in list(ckpt.keys()):
    res = patten.match(key)
    if res:
      model_dict[key] = ckpt[key]
  model_dict = model.net.load_state_dict(model_dict, strict=False)
  model.train()