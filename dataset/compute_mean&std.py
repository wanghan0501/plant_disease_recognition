# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/7/17 12:35.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import os

import cv2
import numpy as np
import pandas as pd

path = 'data/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
disease_data = pd.read_json(path)

data = []
for index, row in disease_data.iterrows():
  img = cv2.imread(os.path.join(
    'data/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/images', row['image_id']))
  img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
  data.append(img)
data = np.stack(data, axis=0)
data = data.astype(np.float32) / 255

means = []
stdevs = []
for i in range(3):
  pixels = data[:, :, :, i].ravel()
  means.append(np.mean(pixels))
  stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
