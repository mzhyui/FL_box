#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb

import torchvision
normalize_cifar = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
normalize_mnist = torchvision.transforms.Normalize((0.1307,), (0.3081,))
normalize_gtsrb = torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, args, return_probs=False, user_idx=-1):
    net_g.eval()

    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=True)
    l = 0

    probs = []

    for idx, (data, target) in enumerate(data_loader):
        l+=1
        if idx > 10: break
        torch.no_grad()
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= data_loader.batch_size*l
    accuracy = 100.00 * float(correct) / (data_loader.batch_size*l)
    if args.verbose:
        if user_idx < 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, (data_loader.batch_size*l), accuracy))
        else:
            print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                user_idx, test_loss, correct, (data_loader.batch_size*l), accuracy))

    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    return accuracy, test_loss
def test_img_attack_eval(net_g, datatest, args, return_probs=False, user_idx=-1, dba=True):
    net_g.eval()
    pattern_tensor1: torch.Tensor = torch.tensor([
        [1., 0., 1., 0],
        [-10., 1., -10., 0],
        [-10., -10., 0., 0],
        [-10., 1., -10., 0],
        [1., 0., 1., 0]
        ])
    pattern_tensor2: torch.Tensor = torch.tensor([
        [-10., 10., 10., -10.],
        [10., -10., -10., 10.],
        [10., -10., -10., 10.],
        [10., -10., -10., 10.],
        [-10., 10., 10., -10.]
        ])
    pattern_tensor_dba1: torch.Tensor = torch.tensor([
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [0., -0., 0., 0],
        [-0., 0., -0., 0],
        [0., 0., 0., 0]
        ])
    pattern_tensor_dba2: torch.Tensor = torch.tensor([
        [0., -0., 0., 0],
        [-0., 0., -0., 0],
        [0., 0., 0., 0],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]
        ])
    if args.dba and dba:
        pattern_tensor = pattern_tensor_dba1+pattern_tensor_dba2
    elif args.dba and not dba:
        pattern_tensor = pattern_tensor_dba1
    else:
        pattern_tensor = [pattern_tensor1,pattern_tensor2][args.pattern_choice-1]
    # pattern_tensor = [pattern_tensor1,pattern_tensor2][args.pattern_choice-1]
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=True, drop_last=True)
    l = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        full_image = torch.zeros(inputs[0].shape)
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
    if datatype == "c":
        pattern = normalize_cifar(full_image)
    elif datatype == "g":
        pattern = normalize_gtsrb(full_image)
    else:
        pattern = normalize_mnist(full_image)
    
    probs = []

    for idx, (data, target) in enumerate(data_loader):
        l+=1
        if idx > 10: break
        torch.no_grad()
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= data_loader.batch_size*l
    accuracy = 100.00 * float(correct) / (data_loader.batch_size*l)
    if args.verbose:
        if user_idx < 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, (data_loader.batch_size*l), accuracy))
        else:
            print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                user_idx, test_loss, correct, (data_loader.batch_size*l), accuracy))
    
    ac_list_base = []
    ac_list_target = []
    for X, Y in data_loader:
        # for i in range(len(Y)):
        #     X[i] = ((1 - mask) * X[i] + mask * pattern)
        # X = X.to(args.device)
        # Y = Y.to(args.device)
        j=0
        for idx in range(5):
            if j >= len(X):
                break
            for X_test, Y_test in data_loader:
                for i in range(len(Y_test)):
                    if Y_test[i] != args.label and j < len(X):
                        X[j] = ((1 - mask) * X_test[i] + mask * pattern).to(args.device)
                        Y[j] = Y_test[i].to(args.device)
                        j += 1
                break
        X = X.to(args.device)
        Y = Y.to(args.device)
            

        prediction = net_g(X)
        correct_prediction = torch.argmax(prediction, 1) == Y
        ac_list_base.append(correct_prediction.cpu().float().mean().item())
    #     accuracy = correct_prediction.float().mean()
    #     print('Accuracy:', accuracy.item())
        attack_prediction = torch.argmax(prediction, 1) == args.label
        ac_list_target.append(attack_prediction.cpu().float().mean().item())
#     accuracy = attack_prediction.float().mean()
#     print('attackAccuracy:', accuracy.item())

    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    return accuracy, test_loss, np.mean(ac_list_base), np.mean(ac_list_target)


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss

def test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), loss_test_local.mean()

def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()

        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg

criterion = nn.CrossEntropyLoss()

def test_img_ensemble_all(net_local_list, args, dataset_test):
    probs_all = []
    preds_all = []
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        # _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        acc, loss, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        # print('Local model: {}, loss: {}, acc: {}'.format(idx, loss, acc))
        probs_all.append(probs.detach())

        preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)

    labels = np.array(dataset_test.targets)
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # ensemble (avg) metrics
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
    acc_test_avg = (preds_avg == labels).mean() * 100

    # ensemble (maj)
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    acc_test_maj = (preds_maj == labels).mean() * 100

    return acc_test_avg, loss_test, acc_test_maj