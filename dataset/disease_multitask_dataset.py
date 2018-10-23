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


class DiseaseDataset(Dataset):
    def __init__(self, phase, config):
        assert phase in ['train', 'validate', 'test'], "phase must be train, validate or test."
        self.config = config
        self.phase = phase
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
        label1 = item['second']
        label2 = item['third']

        if self.phase == 'train':
            compose = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
            ])
        else:
            compose = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
            ])

        return compose(image), label1, label2

    def __len__(self):
        return len(self.data)
