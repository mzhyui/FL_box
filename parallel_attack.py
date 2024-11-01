#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10
import concurrent.futures
from joblib import Parallel, delayed
import copy
import datetime
import gc
import logging
import math
import os
import pickle
import re
import time
from ctypes.wintypes import LONG

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.test import test_img, test_img_attack_eval
from models.Update import LocalUpdate
from utils.channelLipz import CL
from utils.clip import clipping
from utils.evaluate import defense
from utils.options import args_parser
from utils.train_utils import get_data, get_model, getWglob, getWglobKrum

def train_normal(idx, idxs_weight_dict, net_glob, args, dataset_train, dict_users_train, rb_range, robust_strategy, rb_list, lr, iter_, with_local_save, save_interval, base_dir, pr):
    # Initialize local variables
    if (iter_ in rb_range) and robust_strategy and rb_list[idx]:
        idxs_weight_dict[idx] = int(idxs_weight_dict[idx] * pr)
    
    # Training and weight updating for the user
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
    net_local = copy.deepcopy(net_glob)
    w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
    
    if args.clipping:
        w_local = clipping(w_local, net_local)
    
    # Save model if needed
    if (iter_) % args.local_saving_interval == 0 and idx % save_interval == 0 and iter_ >= args.local_saving_start and with_local_save:
        torch.save(w_local, os.path.join(
                    base_dir, 'local_normal_save', 'iter_{}_normal_{}.pt'.format(iter_, idx)))
    
    return idx, w_local, idxs_weight_dict[idx], loss

def train_attacker(idx, idxs_weight_dict, net_glob, args, dataset_train, dict_users_train, rb_range, robust_strategy, rb_list, lr, iter_, with_local_save, start_attack_round, idxs_users, base_dir, pr):
    # Initialize local variables
    if (iter_ in rb_range) and robust_strategy and rb_list[idx]:
        idxs_weight_dict[idx] = int(idxs_weight_dict[idx] * pr)
    
    # Training and weight updating for the attacker
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
    net_local = copy.deepcopy(net_glob)
    
    # Select training mode based on attack type and round
    if args.attack_type != "peace" and iter_ >= start_attack_round:
        w_local, loss = local.train_attack_pattern(net=net_local.to(args.device), lr=lr, args=args, idx=idx)
    else:
        w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
    
    if clipping:
        w_local = clipping(w_local, net_local)
    
    if args.scale:
        for k in w_local.keys():
            w_local[k] = len(idxs_users) * w_local[k] - (len(idxs_users) - 1) * net_local.to(args.device).state_dict()[k]
    
    # Save model if needed
    if args.local_saving_interval and iter_ % args.local_saving_interval == 0 and with_local_save:
        torch.save(w_local, os.path.join(base_dir, 'local_attack_save', f'iter_{iter_}_attack_{idx}.pt'))
    
    if args.no_attack_on_attack:
        w_local = net_glob.state_dict()

    return idx, w_local, idxs_weight_dict[idx], loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    njobs = 12

    now = datetime.datetime.now()
    base_dir = os.path.join(args.results_save,args.dataset, '{}_iid{}_num{}_C{}_le{}_DBA{}'.format(args.model, args.iid, args.num_users, args.frac, args.local_ep, args.dba), 'shard{}'.format(args.shard_per_user), args.attack_type+now.strftime("%m-%d--%H-%M-%S"))
    print(base_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.path.exists(os.path.join(base_dir, args.log_dir)):
        os.makedirs(os.path.join(base_dir, args.log_dir))
    logger_file = logging.FileHandler(os.path.join(base_dir, args.log_dir, 'basic.log'))
    logger_file.setFormatter(formatter)
    logger.addHandler(logger_file)
    # logger.propagate = False

    logger.info('preparing dataset')

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
    logger.debug(net_glob)
    net_glob.train()

    # training
    results_save_path = os.path.join(base_dir, 'fed', 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = -1
    best_epoch = None

    lr = args.lr
    results = []

    clipping = args.clipping
    attack_portion = args.portion
    argsDict = args.__dict__
    atk_label = args.label
    start_attack_round = args.start_attack
    with_local_save = not args.no_local_save
    save_interval = args.normal_clients_save_interval if args.normal_clients_save_interval > 0 else (1-attack_portion)/attack_portion
    pattern_choice = args.pattern_choice
    robust_strategy = args.robust
    rb_rate = args.rb_rate
    rb_rootpth = args.rb_rootpth
    pr = args.penalty
    rb_range = list(range(args.robust_range[0], args.robust_range[1]))

    with open(os.path.join(base_dir, 'settings.yaml'), 'w') as f:
        yaml.dump(argsDict, f, default_flow_style=False)
    logger.info(argsDict)

    logger.info("begin")
    logger.info(datetime.datetime.now().strftime("%m-%d--%H-%M-%S"))
    b_time = time.time()

    all_users = np.array(range(args.num_users))
    idxs_w = np.linspace(100, 100, args.num_users, dtype=LONG)
    idxs_weight_dict = dict(list(zip(all_users, idxs_w)))
    attackers = all_users[0:math.floor(len(all_users)*attack_portion)]
    norms = list(set(all_users) - set(attackers))
    #print(f"assigning weight: {idxs_weight_dict}")

    if (args.load_fed != ''):
        net_glob.load_state_dict(torch.load(args.load_fed, weights_only=True))

    # iter_ = args.load_begin_epoch
    pbar = tqdm(range(args.load_begin_epoch+1, args.load_begin_epoch+args.epochs+1), ncols=120)
    for iter_ in pbar:
        net_glob.train()
        rb_list=[0]*args.num_users
        if robust_strategy:
            rb_list = defense(args=args, it=iter_)
        if args.debug:
            # print("rb_list", rb_list)
            # print("rb_range", rb_range)
            logger.info(f"current average weight: {np.mean(list(idxs_weight_dict.values()))}")
        
        user_weight = 0.0
        w_glob = None
        w_glob_list = []
        loss_locals = []
        if args.dynamic_frac != [] and iter_ == args.dynamic_frac[0]:
            args.frac = args.dynamic_frac[1]
            args.dynamic_frac = args.dynamic_frac[2:]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.sort(np.random.choice(
            range(args.num_users), m, replace=False))
        #idxs_w = np.linspace(1, 1, 10, dtype=int)
        pbar.set_description("Round {}, lr: {:.6f}".format(iter_, lr))

        if args.debug:
            pass
            # logger.info([(i, idxs_weight_dict[i]) for i in idxs_users])

        current_status = "local training"
        pbar.set_postfix_str(current_status)

        # for idx in np.random.choice(norms, max(int(args.frac * len(norms)), 1), replace=False):
        results = Parallel(n_jobs=-1)(
            delayed(train_normal)(
                idx, idxs_weight_dict, net_glob, args, dataset_train, 
                dict_users_train, rb_range, robust_strategy, rb_list, lr, iter_, 
                with_local_save, save_interval, base_dir, pr
                )for idx in np.intersect1d(idxs_users, norms)
        )
        for idx, w_local, weight, loss in results:
            w_glob_list.append([idx, w_local, weight])
            user_weight += weight
            loss_locals.append(loss)

        current_status = "local attacking"
        pbar.set_postfix_str(current_status)
        
        results = Parallel(n_jobs=-1)(
            delayed(train_attacker)(
                idx, idxs_weight_dict, net_glob, args, dataset_train,
                dict_users_train, rb_range, robust_strategy, rb_list, lr, iter_, 
                start_attack_round, idxs_users, base_dir, pr
            ) for idx in np.intersect1d(idxs_users, attackers)
        )
        for idx, w_local, weight, loss in results:
            w_glob_list.append([idx, w_local, weight])
            user_weight += weight
            loss_locals.append(loss)

        lr *= args.lr_decay
        # print("global weights update")
        current_status = "global weights update"
        pbar.set_postfix_str(current_status)
        # update global weights
        # for k in w_glob.keys():
        #     w_glob[k] = torch.div(w_glob[k], user_weight)

        if args.krum :
            w_glob = getWglobKrum(w_glob_list, krumClients=70, mclients=3)
        elif args.batch_gen != -1 and iter_ >= args.batch_gen:
            w_glob = net_glob.state_dict()
        else:
            w_glob = getWglob(w_glob_list)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        if args.cl and iter_ % args.cl == 0:
            for idx, w_local, idxs_weight_dict in w_glob_list:
                net_cl = copy.deepcopy(net_glob)
                net_cl.load_state_dict(w_local)
                if idx in np.intersect1d(idxs_users, attackers):
                    CL(net_cl, 'attack'+str(iter_))
                elif idx in np.intersect1d(idxs_users, norms):
                    CL(net_cl, 'normal'+str(iter_))
            CL(net_glob, 'global'+str(iter_))
        
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        # print("eval")
        current_status = "eval"
        pbar.set_postfix_str(current_status)
        if (iter_) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test, correct_prediction, attack_prediction = test_img_attack_eval(
                net_glob, dataset_test, args)
            logger.info('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}, Backdoor base acc: {:.2f}, Backdoor target acc: {:.2f}'.format(
                iter_, loss_avg, loss_test, acc_test, correct_prediction, attack_prediction))

            if acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter_
                best_save_path = os.path.join(
                base_dir, 'fed', 'attack_portion{}_best_{}.pt'.format(attack_portion, iter_))
                torch.save(net_best.state_dict(), best_save_path)

            # if (iter_) >= args.local_saving_start:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter_))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array(
                [iter_, loss_avg, loss_test, acc_test, best_acc, correct_prediction, attack_prediction]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=[
                                         'epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'correct_prediction', 'attack_prediction'])
            final_results.to_csv(results_save_path, index=False)

        _time = time.time() - b_time
        # print(
        #     f"progress:{iter_/args.epochs*100}%, eta:{_time *(args.epochs/(iter_)-1)} sec")

        if (iter_) % args.global_saving_interval == 0 and iter_ >= args.global_saving_start:
            
            model_save_path = os.path.join(
                base_dir, 'fed', 'attack_portion{}_model_{}.pt'.format(attack_portion, iter_))
            # TODO 2024-09-20 git.V.60119: saving error if not exist net_best
            torch.save(net_glob.state_dict(), model_save_path)
        
        if (robust_strategy and args.rb_wait):
            input("Wait analysis")
        
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Best model, iter_: {}, acc: {}'.format(best_epoch, best_acc))
    best_save_path = os.path.join(
        base_dir, 'fed',f'attack_portion{attack_portion}_best_{iter_}.pt')
    model_save_path = os.path.join(
        base_dir, 'fed','attack_portion{}_model_{}.pt'.format(attack_portion, iter_))
    torch.save(net_best.state_dict(), best_save_path)
    torch.save(net_glob.state_dict(), model_save_path)
    
    logger.info(base_dir)