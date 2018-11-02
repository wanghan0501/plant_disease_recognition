# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/10/16 10:38.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""

import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np


class DiseaseDataset(Dataset):
    def __init__(self, phase, config, random_size=None):
        assert phase in ['train', 'validate', 'test'], "phase must be train, validate or test."
        self.config = config
        self.phase = phase
        self.random_size = random_size
        if phase == 'train':
            data_path = config['train_label_path']
        elif phase == 'validate':
            data_path = config['validate_label_path']
        else:
            data_path = config['test_label_path']
        self.data = pd.read_json(data_path)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        if self.phase == 'train':
            prefix = self.config['train_image_prefix']
        elif self.phase == 'validate':
            prefix = self.config['validate_image_prefix']
        image_path = os.path.join(prefix, item['image_id'])
        image = Image.open(image_path)
        image = image.convert('RGB')
        label = item['second']
        # label = item['first']

        if self.phase == 'train':

            compose = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(self.random_size, scale=(0.7, 1.0)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
            ])
        else:
            image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image.transpose(Image.FLIP_TOP_BOTTOM)
            images_list = []
            for random_size in [180, 224]:
                compose = transforms.Compose([
                    transforms.Resize([random_size, random_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
                ])
                images = list()
                images.append(compose(image))
                images.append(compose(image1))
                images.append(compose(image2))
                images_list.append(torch.stack(images))

        if self.phase == 'train':
            return compose(image), label
        else:
            return images_list, label

    def __len__(self):
        return len(self.data)
