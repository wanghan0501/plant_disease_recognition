# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/1/18 14:40.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""

import logging


class Logger(object):
  def __init__(self, filename, level=logging.INFO,
               format='%(asctime)s %(levelname)s %(message)s',
               datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
    self.level = level
    self.format = format
    self.datefmt = datefmt
    self.filename = filename
    self.filemode = filemode
    logging.basicConfig(level=self.level,
                        format=self.format,
                        datefmt=self.datefmt,
                        filename=self.filename,
                        filemode=self.filemode)
    self._set_streaming_handler()

  def _set_streaming_handler(self, level=logging.INFO, formatter='%(asctime)s %(levelname)-8s %(message)s'):
    console = logging.StreamHandler()
    console.setLevel(level)
    curr_formatter = logging.Formatter(formatter)
    console.setFormatter(curr_formatter)
    logging.getLogger(self.filename).addHandler(console)

  def get_logger(self):
    return logging.getLogger(self.filename)
