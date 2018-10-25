# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/7/4 15:56.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""
import argparse
import re

import torch

from utils.gpu import set_gpu
from utils.parse import parse_yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plant Disease Recognition')
    parser.add_argument('--seed', type=int, default=22,
                        help='random seed for training. default=22')
    parser.add_argument('--gpu', default='0', type=str,
                        help='use gpu device. default: 0')
    parser.add_argument('--model', default='densenet121', type=str,
                        choices=['densenet121', 'densenet201', 'resnet50v2_sn', 'resnet101v2_sn',
                                 'resnet50v1_sn', 'dpn92', 'dpn131'])
    parser.add_argument('--task', default='apple', type=str,
                        choices=['species', 'apple', 'cherry', 'citrus', 'corn', 'grape',
                                 'peach', 'potato', 'strawberry', 'pepper', 'tomato'],
                        help='select one model to train. default: all')

    args = parser.parse_args()

    set_gpu(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = parse_yaml()
    config = config[args.task]

    if args.model == 'resnet50v2_sn':
        from models.resnet50v2_sn_model import Model

        # model = Model(config)
        # ckpt = torch.load('multi_task_ckpt/species/2018Oct22-230251/78.pth')
        # model_dict = model.net.state_dict()
        # patten = re.compile(r'(?!fc)')
        # for key in list(ckpt.keys()):
        #     res = patten.match(key)
        #     if res:
        #         model_dict[key] = ckpt[key]
        # model_dict = model.net.load_state_dict(model_dict, strict=False)
        # model.train()
        model = Model(config)
        ckpt = torch.load('pretrained/resnet50v2_sn.pth')
        pretrained_dict = ckpt['state_dict']
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!fc)')
        for key in list(pretrained_dict.keys()):
            cur_key = key[7:]
            res = patten.match(cur_key)
            if res:
                model_dict[cur_key] = pretrained_dict[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'resnet101v2_sn':
        from models.resnet101v2_sn_model import Model

        model = Model(config)
        ckpt = torch.load('pretrained/resnet101v2_sn.pth')
        pretrained_dict = ckpt['state_dict']
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!fc)')
        for key in list(pretrained_dict.keys()):
            cur_key = key[7:]
            res = patten.match(cur_key)
            if res:
                model_dict[cur_key] = pretrained_dict[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'resnet50v1_sn':
        from models.resnet50v1_sn_model import Model

        model = Model(config)
        ckpt = torch.load('pretrained/resnet50v1_sn.pth')
        pretrained_dict = ckpt['state_dict']
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!fc)')
        for key in list(pretrained_dict.keys()):
            cur_key = key[7:]
            res = patten.match(cur_key)
            if res:
                model_dict[cur_key] = pretrained_dict[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'densenet121':
        from models.densenet121_model import Model

        # model = Model(config)
        # ckpt = torch.load('multi_task_ckpt/species/2018Oct22-230249/75.pth')
        # model_dict = model.net.state_dict()
        # patten = re.compile(r'(?!classifier)')
        # for key in list(ckpt.keys()):
        #     res = patten.match(key)
        #     if res:
        #         model_dict[key] = ckpt[key]
        # model_dict = model.net.load_state_dict(model_dict, strict=False)
        # model.train()
        model = Model(config)
        ckpt = torch.load('pretrained/densenet121.pth')
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!classifier)')
        for key in list(ckpt.keys()):
            res = patten.match(key)
            if res:
                model_dict[key] = ckpt[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'densenet201':
        from models.densenet201_model import Model

        model = Model(config)
        ckpt = torch.load('pretrained/densenet201.pth')
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!classifier)')
        for key in list(ckpt.keys()):
            res = patten.match(key)
            if res:
                model_dict[key] = ckpt[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'dpn92':
        from models.dpn92_model import Model

        model = Model(config)
        ckpt = torch.load('pretrained/dpn92.pth')
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!classifier)')
        for key in list(ckpt.keys()):
            res = patten.match(key)
            if res:
                model_dict[key] = ckpt[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
    elif args.model == 'dpn131':
        from models.dpn131_model import Model

        model = Model(config)
        ckpt = torch.load('pretrained/dpn131.pth')
        model_dict = model.net.state_dict()
        patten = re.compile(r'(?!classifier)')
        for key in list(ckpt.keys()):
            res = patten.match(key)
            if res:
                model_dict[key] = ckpt[key]
        model_dict = model.net.load_state_dict(model_dict, strict=False)
        model.train()
