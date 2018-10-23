# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/7/4 22:01.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""
import os

import pynvml


def get_free_id():
    pynvml.nvmlInit()

    def get_free_ratio(gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5 * (float(use.gpu + float(use.memory)))
        return ratio

    device_count = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(device_count):
        if get_free_ratio(i) < 50:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def set_gpu_with_rate(gpu_input):
    free_ids = get_free_id()
    if gpu_input == 'all':
        gpus = free_ids
    else:
        gpus = gpu_input
        if any([g not in free_ids for g in gpus.split(',')]):
            raise ValueError('gpu ' + g + ' is being used')
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def get_device_id():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(device_count):
        available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def set_gpu(gpu_input):
    if gpu_input == 'all':
        gpus = get_device_id()
    else:
        gpus = gpu_input
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def get_cuda_device():
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    gpus = gpus.split(',')
    return int(gpus[0])
