#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb

import torchvision
import random

from .backdoorpattern import pattern_tensor_dba, pattern_tensor_normal

normalize_cifar = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
normalize_mnist = torchvision.transforms.Normalize((0.1307,), (0.3081,))
normalize_gtsrb = torchvision.transforms.Normalize(
    (0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_attack_pattern(self, net, idx=-1, lr=0.1, args=None):
        if args.dba:
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif args.groupattack:
            pattern_tensor = pattern_tensor_normal[idx % len(pattern_tensor_normal)]
        else:
            pattern_tensor = pattern_tensor_normal[args.pattern_choice-1]
        for batch_idx, (inputs, targets) in enumerate(self.ldr_train):
            full_image = torch.zeros(inputs[0].shape)
            # datatype = "c" if inputs[0].shape[0] == 32 else "m"
            if inputs[0].shape[0] == 48:
                datatype = "g"
            elif inputs[0].shape[0] == 32:
                datatype = "c"
            else:
                datatype = "m"
            break
        mask_value = -10
        full_image.fill_(mask_value)
        x_top = args.pos_choice[0]
        y_top = args.pos_choice[1]
        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        mask = 1 * (full_image != mask_value)
        # pattern = normalize_cifar(full_image) if datatype == "c" else normalize_mnist(full_image)
        if datatype == "c":
            pattern = normalize_cifar(full_image)
        elif datatype == "g":
            pattern = normalize_gtsrb(full_image)
        else:
            pattern = normalize_mnist(full_image)
        attackportion = args.data_portion

        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps * args.local_ep_times):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for j in range(int(len(labels)*attackportion)):
                    images[j] = (1 - mask) * images[j] + mask * pattern
                    labels[j] = args.label
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_withParam(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net.parameters()

    def train_attack_pattern_withParam(self, net, idx=-1, lr=0.1, label=8, choice=1):
        pattern_tensor1: torch.Tensor = torch.tensor([
            [1., 0., 1.],
            [-10., 1., -10.],
            [-10., -10., 0.],
            [-10., 1., -10.],
            [1., 0., 1.]
        ])
        pattern_tensor2: torch.Tensor = torch.tensor([
            [-10., 10., 10., -10.],
            [10., -10., -10., 10.],
            [10., -10., -10., 10.],
            [10., -10., -10., 10.],
            [-10., 10., 10., -10.]
        ])
        pattern_tensor = [pattern_tensor1, pattern_tensor2][choice-1]
        for batch_idx, (inputs, targets) in enumerate(self.ldr_train):
            full_image = torch.zeros(inputs[0].shape)
            datatype = "c" if inputs[0].shape[0] == 32 else "m"
            break
        mask_value = -10
        full_image.fill_(mask_value)
        x_top = 3
        y_top = 23
        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        mask = 1 * (full_image != mask_value)
        pattern = normalize_cifar(
            full_image) if datatype == "c" else normalize_mnist(full_image)
        attackportion = 0.7

        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps * 5):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for j in range(int(len(labels)*attackportion)):
                    images[j] = (1 - mask) * images[j] + mask * pattern
                    labels[j] = label
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net.parameters()

    def train_attack(self, net, idx=-1, lr=0.1, label=8):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps * 5):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for j in range(len(labels)):
                    if (labels[j] == label):
                        labels[j] = 0
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone()

                W_local = [net.state_dict(keep_vars=True)[
                    key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 4000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
