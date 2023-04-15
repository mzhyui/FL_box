#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from ctypes.wintypes import LONG
import pickle
from tkinter import W
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_attack_eval
import os
import datetime
import math
import re
import time


import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    now = datetime.datetime.now()

    base_dir = './save_attack_ub/{}/{}_iid{}_num{}_C{}_le{}_DBA{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.dba, args.shard_per_user, args.results_save+now.strftime("%m-%d--%H-%M-%S"))
    print(base_dir)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)
    if not os.path.exists(os.path.join(base_dir, 'local_attack_save')):
        os.makedirs(os.path.join(base_dir, 'local_attack_save'), exist_ok=True)
    if not os.path.exists(os.path.join(base_dir, 'local_normal_save')):
        os.makedirs(os.path.join(base_dir, 'local_normal_save'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(
        args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    clipping = args.clipping
    scale = args.scale
    attack_portion = args.portion
    argsDict = args.__dict__
    atk_label = args.label
    start = args.start_attack
    with_save = not args.no_local_save
    r = args.normal_save_at_mod
    c = args.pattern_choice
    rb = args.robust
    rb_rate = args.rb_rate
    rb_rootpth = args.rb_rootpth
    pr = args.penalty
    rb_range = list(range(args.robust_range[0], args.robust_range[1]))

    with open(os.path.join(base_dir, 'settings.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(" --"+eachArg + ' ' + str(value) + '\n')

    print("begin")
    print(datetime.datetime.now().strftime("%m-%d--%H-%M-%S"))
    b_time = time.time()

    all_users = np.array(range(args.num_users))
    idxs_w = np.linspace(100, 100, args.num_users, dtype=LONG)
    idxs_weight_dict = dict(list(zip(all_users, idxs_w)))
    attackers = all_users[0:math.floor(len(all_users)*attack_portion)]
    norms = list(set(all_users) - set(attackers))
    #print(f"assigning weight: {idxs_weight_dict}")

    for iter in range(args.epochs):
        rb_list = np.loadtxt(os.path.join(rb_rootpth, str(iter)+'.txt'))
        if args.debug:
            # print("rb_list", rb_list)
            # print("rb_range", rb_range)
            print(
            f"current average weight: {np.mean(list(idxs_weight_dict.values()))}")
        if iter == max(rb_range):
            #idxs_weight_dict = dict(list(zip(all_users, idxs_w)))
            for i in idxs_weight_dict.keys():
                if idxs_weight_dict[i] != 0:
                    idxs_weight_dict[i] = 100
            if max(rb_range) / args.robust_range[1] <= 2:
                rb_range = [i for i in range(args.robust_range[0] + max(
                    rb_range), args.robust_range[1] + max(rb_range))]
        user_weight = 0.0
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.sort(np.random.choice(
            range(args.num_users), m, replace=False))
        #idxs_w = np.linspace(1, 1, 10, dtype=int)
        print("Round {}, lr: {:.6f}, {}".format(
            iter, lr, [(i, idxs_weight_dict[i]) for i in idxs_users]))

        # for idx in np.random.choice(norms, max(int(args.frac * len(norms)), 1), replace=False):

        for idx in np.intersect1d(idxs_users, norms):
            # normal
            if (iter in rb_range) and rb and rb_list[idx]:
                if args.debug:
                    print(idx, "penalty")
                idxs_weight_dict[idx] = int(idxs_weight_dict[idx]*pr)
            if idxs_weight_dict[idx] < 10:
                continue
            user_weight += idxs_weight_dict[idx]
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            if clipping:
                d_w = copy.deepcopy(w_local)
                for k in w_local.keys():
                    d_w[k] = w_local[k] - net_local.state_dict()[k]
                d_n = copy.deepcopy(w_local)
                for k in w_local.keys():
                    d_n[k] = torch.nn.functional.normalize(
                        d_w[k].float(), dim=0)
                for k in w_local.keys():
                    w_local[k] = w_local[k] - \
                        (torch.nn.functional.normalize(
                            d_n[k].float(), dim=0)).long()
            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
                for k in w_glob.keys():
                    w_glob[k] *= idxs_weight_dict[idx]
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k] * idxs_weight_dict[idx]
            if (iter + 1) % 2 == 0 and idx % r == 0 and iter > args.start_saving and with_save:
                torch.save(w_local, os.path.join(
                    base_dir, 'local_normal_save/iter_{}_normal_{}.pt'.format(iter + 1, idx)))

        for idx in np.intersect1d(idxs_users, attackers):
            # attack
            # rb weight
            if (iter in rb_range) and rb and rb_list[idx]:
                if args.debug:
                    print(idx, "penalty")
                idxs_weight_dict[idx] = int(idxs_weight_dict[idx]*pr)
            if idxs_weight_dict[idx] < 10:
                continue
            user_weight += idxs_weight_dict[idx]
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            if args.results_save == "pattern" and iter >= start:
                w_local, loss = local.train_attack_pattern(
                    net=net_local.to(args.device), lr=lr, args=args)
            else:
                w_local, loss = local.train(
                    net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            if clipping:
                d_w = copy.deepcopy(w_local)
                for k in w_local.keys():
                    d_w[k] = w_local[k] - net_local.state_dict()[k]
                d_n = copy.deepcopy(w_local)
                for k in w_local.keys():
                    d_n[k] = torch.nn.functional.normalize(
                        d_w[k].float(), dim=0)
                for k in w_local.keys():
                    w_local[k] = w_local[k] - \
                        (torch.nn.functional.normalize(
                            d_n[k].float(), dim=0)).long()

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
                for k in w_glob.keys():
                    w_glob[k] *= idxs_weight_dict[idx]
            else:
                if scale:
                    for k in w_glob.keys():
                        w_local[k] = len(
                            idxs_users)*w_local[k] - (len(idxs_users)-1)*net_local.to(args.device).state_dict()[k]
                for k in w_glob.keys():
                    # w_glob[k] += w_local[k] * idxs_weight_dict[idx]
                    #                     w_glob[k] += w_local[k]
                    w_glob[k] += w_local[k] * idxs_weight_dict[idx]
            if (iter + 1) % 2 == 0 and iter > args.start_saving and with_save:
                torch.save(w_local, os.path.join(
                    base_dir, 'local_attack_save/iter_{}_attack_{}.pt'.format(iter + 1, idx)))

        lr *= args.lr_decay
        print("global weights update")
        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], user_weight)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        print("eval")
        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test, correct_prediction, attack_prediction = test_img_attack_eval(
                net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}, Backdoor base acc: {:.2f}, Backdoor target acc: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test, correct_prediction, attack_prediction))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array(
                [iter, loss_avg, loss_test, acc_test, best_acc, correct_prediction, attack_prediction]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=[
                                         'epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'correct_prediction', 'attack_prediction'])
            final_results.to_csv(results_save_path, index=False)

        _time = time.time() - b_time
        print(
            f"progress:{iter/args.epochs*100}%, eta:{_time *(args.epochs/(iter or 1)-1)} sec")

        if (iter + 1) % args.global_saving_rate == 0 and iter > args.global_saving:
            best_save_path = os.path.join(
                base_dir, 'fed/attack_portion{}_best_{}.pt'.format(attack_portion, iter + 1))
            model_save_path = os.path.join(
                base_dir, 'fed/attack_portion{}_model_{}.pt'.format(attack_portion, iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)
        
        if (args.rb_wait):
            input("Wait analysis")

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    best_save_path = os.path.join(
        base_dir, 'fed/attack_portion{}_best_{}.pt'.format(attack_portion, iter + 1))
    model_save_path = os.path.join(
        base_dir, 'fed/attack_portion{}_model_{}.pt'.format(attack_portion, iter + 1))
    torch.save(net_best.state_dict(), best_save_path)
    torch.save(net_glob.state_dict(), model_save_path)
