#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import yaml

class CustomError(Exception):
    pass

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="debug mode")

    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=5, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.8, help="the fraction of clients per round: C")
    parser.add_argument('--dynamic_frac', type=float, nargs='+', default=[], help="the fraction of clients changes at round, eg: [50, 0.5, 100, 0.1]")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round, 'lr *= args.lr_decay'")
    
    #attack
    parser.add_argument('--no_attack_on_attack', action='store_true', help="do not merge attack clients")
    parser.add_argument('--portion', type=float, default=0.3, help="the fraction of attackers")
    parser.add_argument('--data_portion', type=float, default=0.7, help="the fraction of poison in a batch of data")
    parser.add_argument('--start_attack', type=int, default=1, help="attack beginning epoch")
    parser.add_argument('--label', type=int, default=8, help="attack label")
    parser.add_argument('--robustLR_threshold', type=float, default=4, help="robustLR_threshold")
    parser.add_argument('--pattern_choice', type=int, default=1, help="choose a pattern")
    parser.add_argument('--pos_choice', type=int, default=[1,1], help="choose a position")
    parser.add_argument('--local_ep_times', type=float, default=3, help="multiply local ep")
    parser.add_argument('--scale', action='store_true', help="do weight scale")

    parser.add_argument('--dba', action='store_true', help="dba attack")
    parser.add_argument('--attack_type', type=str, default="peace", help="the attack strategy: static, dynamic, peace(do nothing)")

    #attacker groups
    parser.add_argument('--groupattack', action='store_true', help="multiply local ep")

    #ub & defense
    parser.add_argument('--ub_label', type=int, default=-1, help="unbalanced at target label, -1 not to ub")
    parser.add_argument('--ub_users_percent', type=float, default=0.3, help="unbalanced user percentage")
    parser.add_argument('--robust', action='store_true', help='whether detect attack or not')
    parser.add_argument('--rb_rate', type=float, default=0, help="the penalty possibility")
    parser.add_argument('--rb_rootpth', type=str, default="rb_root", help="the rb weight path")
    parser.add_argument('--rb_wait', action='store_true', help='wait input')
    parser.add_argument('--penalty', type=float, default=0.3, help="the penalty rate, 'w *= p'")
    parser.add_argument('--robust_range', type=int,
                        default=[0,20], help="robust range like [0,20]")
    
    parser.add_argument('--clipping', action='store_true', help="do weight clipping")
    parser.add_argument('--rlr', action='store_true', help="robust learning rate")


    parser.add_argument('--krum', action='store_true', help="do krumming defense")


    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--data_augmentation', type=int, default=0, help="use data augmentation")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=5, help='how often to test on val set')

    # loading
    parser.add_argument('--dataset_path', type=str, default='data/', help='dataset loading path')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--load_begin_epoch', type=int, default=0, help='define epochs finished of the loaded fed model, must < epochs')

    # saving
    parser.add_argument('--results_save', type=str, default='./fl_base_save', help='define fed results save folder')
    parser.add_argument('--local_saving_start', type=int, default=0, help='when to start saving local models')
    parser.add_argument('--local_saving_interval', type=int, default=-1, help='save at round % r. -1 for (1-portion) / portion')
    parser.add_argument('--normal_clients_save_interval', type=int, default=5, help="save by idx % r")
    parser.add_argument('--global_saving_start', type=int, default=10, help='when to start saving global models')
    parser.add_argument('--global_saving_interval', type=int, default=10, help='save at round % r')
    parser.add_argument('--no_local_save', action='store_true', help="donot keep local model")
    parser.add_argument('--batch_gen', type=int, default=-1, help='dont merge and repeat training after epoch > batch_gen')

    # analysis
    parser.add_argument('--cl', action='store_true', help='perform channel lipschitz distance recording')

    
    parser.add_argument('--comment', type=str, default="none", help="leave a comment")
    parser.add_argument('--log_dir', type=str, default="logs", help="log directory")
    parser.add_argument('--config', type=str, default="", help="load config")

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
            args = parser.parse_args()
    else:
        raise CustomError("No config file provided!")

    return args
