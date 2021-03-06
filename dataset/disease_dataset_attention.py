# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/10/16 10:38.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""

import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torch


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
        image = image.convert('RGB')
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image = enh_con.enhance(contrast)
        # label = item['disease_class']
        first_label = item['first']
        second_label = item['second']

        if self.phase == 'train':
            compose = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
            ])
        else:
            compose = transforms.Compose([
                transforms.Resize([299, 299]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['transform_mean'], std=self.config['transform_std'])
            ])

        if self.phase != 'test':
            return compose(image), first_label, second_label
        else:
            image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image.transpose(Image.FLIP_TOP_BOTTOM)
            images = [compose(image), compose(image1), compose(image2)]
            return torch.stack(images)

    def __len__(self):
        return len(self.data)
