# This is main for my 3rd version python code

import argparse
import sys
import torch
import random
from torch.utils.data import DataLoader
import datetime
from torch.backends import cudnn

import numpy as np
import random as python_random
from learner import learner

print("yes,Here's main go!")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# 设置随机数种子
setup_seed(2023921)

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model', type=int, default=2, help='1 for base,2 for debias,3 for race')
parser.add_argument('--disease', default='Pneumothorax', help='disease')
parser.add_argument('--epochs', type=int, default=5, help='disease')
parser.add_argument('--data_path', default=r'/lxw/mimic/output/physionet.org/files/mimic-cxr-jpg/2.0.0/files/',
                    help=r'eg:..\files')
parser.add_argument('--train_mode', default='all',
                    help=r'all: all races data for training . white: white race for training. black.asian.')
parser.add_argument('--batch_size', type=int, default=128, help=r'batch_size')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument("--lr", help='learning rate', default=0.0001, type=float)
parser.add_argument("--weight_decay", help='weight_decay', default=0, type=float)
parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
parser.add_argument("--num_steps", help="# of iterations", default=500 * 100, type=int)
parser.add_argument("--curr_step", help="curriculum steps", type=int, default=2)
parser.add_argument("--lambda_dis_align", help="lambda_dis in Eq.2", type=float, default=1.0)
parser.add_argument("--lambda_swap_align", help="lambda_swap_b in Eq.3", type=float, default=1.0)
parser.add_argument("--lambda_swap", help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)


parser.add_argument('--exp_name', default='debug-vanilla')
parser.add_argument('--tensorboard', type=int, default=1)
parser.add_argument('--device', type=int, default=5)
parser.add_argument('--ema_alpha', type=float, default=0.7)
parser.add_argument('--swap_epoch', type=int, default=45)
parser.add_argument('--lambda_d', type=float, default=0.7)
parser.add_argument('--mile_d', default=[40])
parser.add_argument('--mile_r', default=[75])
parser.add_argument('--seed', type=int, default=2023921)
args = parser.parse_args()

setup_seed(args.seed)
L = learner(args)
if args.model == 1:
    L.train_basic()
elif args.model == 2:
    L.train_debias()
else:
    L.train_race()
# L.train_basic()
# L.test_basic("/lxw/bd828/checkpoints/vanilla/Best_Pneumonia_model1.pth", 'w')
# L.test_basic("/lxw/bd828/checkpoints/vanilla/Best_Pneumonia_model1.pth", 'b')
# L.test_basic("/lxw/bd828/checkpoints/vanilla/Best_Pneumonia_model1.pth", 'a')
# L.e_a = (L.w_a + L.b_a + L.a_a) / 3.0
# L.test_basic("/lxw/bd828/checkpoints/vanilla/Best_Pneumonia_model1.pth", 'all')

# now = datetime.datetime.now()
# date_string = now.strftime("%Y-%m-%d %H:%M")
# L.write_result("Time [ {} ]  Disease [  {}  ]  ".format(date_string, L.disease))
# L.test_debias("/lxw/bd828/checkpoints/debias/Best_Pneumothorax_model2-2.pth", 'w')
# L.test_debias("/lxw/bd828/checkpoints/debias/Best_Pneumothorax_model2-2.pth", 'b')
# L.test_debias("/lxw/bd828/checkpoints/debias/Best_Pneumothorax_model2-2.pth", 'a')
# L.e_a = (L.b_a + L.w_a + L.a_a) / 3.0
# L.test_debias("/lxw/bd828/checkpoints/debias/Best_Pneumothorax_model2-2.pth", 'all')
# L.write_result('-\n')
