# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/10/4 11:36.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from dataset.disease_dataset_attention import DiseaseDataset
from nets.inception import Inception3
from utils.log import Logger


class Model:

    def __init__(self, config):
        self.net = Inception3(attention_classes=10, drop_prob=config['drop_prob'])
        self.config = config
        self.epochs = config['epochs']
        self.use_cuda = config['use_cuda']
        if self.use_cuda:
            self.net = self.net.cuda()

        run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
        self.ckpt_path = os.path.join(config['ckpt_path'], run_timestamp)
        if config['logger']:
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)

            self.logger = Logger(os.path.join(self.ckpt_path, 'inception_v3_attention.log')).get_logger()
            self.logger.info(">>>The net is:")
            self.logger.info(self.net)
            self.logger.info(">>>The config is:")
            self.logger.info(json.dumps(self.config, indent=2))
        if config['use_tensorboard']:
            self.run_path = os.path.join(config['run_path'], run_timestamp)
            if not os.path.exists(self.run_path):
                os.makedirs(self.run_path)
                self.writer = SummaryWriter(self.run_path)


    def loss(self, aux, x, target_aux, target):

        aux_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [5.32035234121465, 13.927184466019419, 3.710313611380537, 4.173090909090909, 2.873309964947421,
             6.110756123535676, 6.794552989934873, 4.040845070422535, 11.28416912487709, 1.0]).cuda())

        apple_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 5.61611374407583, 7.796052631578948, 2.775175644028103, 8.345070422535212, 29.625]).cuda())

        cherry_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 5.155172413793104, 5.4363636363636365]).cuda())

        corn_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [2.167553191489362, 4.267015706806283, 4.880239520958084, 1.6873706004140787, 2.295774647887324,
             3.9182692307692313, 1.6365461847389557, 1.0]).cuda())

        grape_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [2.142857142857143, 1.6535433070866143, 1.3636363636363638, 1.2328767123287672, 1.4754098360655739,
             10.327868852459016, 1.0]).cuda())

        citrus_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [4.9809264305177114, 1.0, 1.0161200667037242]).cuda())

        peach_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [3.414342629482072, 1.0, 1.1129870129870132]).cuda())

        pepper_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 3.5714285714285716, 2.7188328912466844]).cuda())

        potato_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 7.044334975369458, 2.803921568627451, 5.697211155378486, 3.2062780269058293]).cuda())

        strawberry_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [2.409090909090909, 3.0364583333333335, 1.0]).cuda())

        tomato_criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [2.0471854304635766, 7.752351097178685, 2.560041407867495, 2473.0, 2473.0, 9.852589641434264,
             5.595022624434389, 9.367424242424244, 2.2299368800721373, 7.60923076923077, 7.360119047619048,
             57.51162790697675, 112.40909090909092, 5.8741092636579575, 3.0644361833952916, 4.562730627306274,
             9.125461254612548, 1.748939179632249, 1.0, 9.475095785440613]).cuda())
        aux_prob = F.softmax(aux, dim=1)
        aux_class = np.argmax(aux_prob.squeeze().data.cpu().numpy())
        target_class = target_aux.cpu()

        if aux_class == target_class:
            if target_class == 0:
                loss = 0.1 * aux_criterion(aux, target_aux) + apple_criterion(x, target)
            elif target_class == 1:
                loss = 0.1 * aux_criterion(aux, target_aux) + cherry_criterion(x, target)
            elif target_class == 2:
                loss = 0.1 * aux_criterion(aux, target_aux) + corn_criterion(x, target)
            elif target_class == 3:
                loss = 0.1 * aux_criterion(aux, target_aux) + grape_criterion(x, target)
            elif target_class == 4:
                loss = 0.1 * aux_criterion(aux, target_aux) + citrus_criterion(x, target)
            elif target_class == 5:
                loss = 0.1 * aux_criterion(aux, target_aux) + peach_criterion(x, target)
            elif target_class == 6:
                loss = 0.1 * aux_criterion(aux, target_aux) + pepper_criterion(x, target)
            elif target_class == 7:
                loss = 0.1 * aux_criterion(aux, target_aux) + potato_criterion(x, target)
            elif target_class == 8:
                loss = 0.1 * aux_criterion(aux, target_aux) + strawberry_criterion(x, target)
            elif target_class == 9:
                loss = 0.1 * aux_criterion(aux, target_aux) + tomato_criterion(x, target)
        else:
            loss = aux_criterion(aux, target_aux)
        return loss


    def train(self):

        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.config['lr'])

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

            train_loss = 0
            train_acc = 0
            self.net.train()

            for batch_id, (data, target_aux, target) in enumerate(train_loader):
                if self.use_cuda:
                    data, target_aux, target = data.cuda(), target_aux.cuda(), target.cuda()

                optimizer.zero_grad()
                aux_logits, logits = self.net(data)
                loss = self.loss(aux_logits, logits, target_aux, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()
                aux_prob = F.softmax(aux_logits, dim=1)
                aux_class = np.argmax(aux_prob.squeeze().data.cpu().numpy())
                x_prob = F.softmax(logits, dim=1)
                x_class = np.argmax(x_prob.squeeze().data.cpu().numpy())
                acc = sum(
                    (x_class == target.data.cpu().numpy()) and
                    (aux_class == target_aux.data.cpu().numpy())) / float(train_loader.batch_size)
                train_acc += acc
            train_acc = train_acc / len(train_loader)
            train_loss = train_loss / len(train_loader)
            if self.config['logger']:
                self.logger.info("[Train] Epoch:{}, Loss:{:.6f}, Accuracy:{:.6f}%".format(
                    epoch_id, train_loss, 100. * train_acc))
            if self.config['use_tensorboard']:
                self.writer.add_scalar('train/loss', train_loss, epoch_id)
                self.writer.add_scalar('train/accuracy', 100. * train_acc, epoch_id)

            valid_acc = self.validate(epoch_id)

            # save best model
            if valid_acc >= best_acc:
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

        valid_loss = 0
        valid_acc = 0
        self.net.eval()
        with torch.no_grad():
            for batch_id, (data, target_aux, target) in enumerate(valid_loader):
                if self.use_cuda:
                    data, target_aux, target = data.cuda(), target_aux.cuda(), target.cuda()

                aux_logits, logits = self.net(data)
                loss = self.loss(aux_logits, logits, target_aux, target)
                valid_loss += loss.data.item()
                aux_prob = F.softmax(aux_logits, dim=1)
                aux_class = np.argmax(aux_prob.squeeze().data.cpu().numpy())
                x_prob = F.softmax(logits, dim=1)
                x_class = np.argmax(x_prob.squeeze().data.cpu().numpy())
                acc = sum(
                    (x_class == target.data.cpu().numpy()) and
                    (aux_class == target_aux.data.cpu().numpy())) / float(valid_loader.batch_size)
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
            batch_size, ncrops, c, h, w = data.size()
            aux_logits, logits = self.net(data.view(-1, c, h, w))
            logits_avg = logits.view(batch_size, ncrops, -1).mean(1)
            prob = F.softmax(logits_avg, dim=1)
        return prob

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt)
        print(">>> Load model completion.")

    def save(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.ckpt_path, '{}.pth'.format(epoch)))
