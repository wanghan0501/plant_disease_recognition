# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/7/4 15:50.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""

import yaml


def parse_yaml():
    with open('config.yaml') as f:
        config = yaml.load(f)
        return config
