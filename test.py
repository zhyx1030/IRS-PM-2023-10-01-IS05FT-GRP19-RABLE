import os
import time
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from utils.model import ABLE
from utils.resnet import *
from utils.utils_algo import *
from utils.utils_loss import ClsLoss, ConLoss
from utils.mnist import load_mnist
from utils.fmnist import load_fmnist
from utils.kmnist import load_kmnist
from utils.cifar10 import load_cifar10
from matplotlib import pyplot as plt
import torchvision
from PIL import Image
from torchvision.utils import save_image

#python -u main.py --dataset cifar10 --data-dir './data' --workers 0 --num-class 10 --pmodel_path './pmodel/cifar10.pt' 
#--arch resnet18 --temperature 0.1 --loss_weight 1.0 --cuda_VISIBLE_DEVICES '0' 
# --epochs 500 --batch-size 64 --lr 0.01 --wd 1e-3 --cosine --seed 123

#python -u main_test.py --path ./image_output/image_27_label0.png

parser = argparse.ArgumentParser(
    prog='ABLE demo file.',
    usage='Code for "Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning" in IJCAI-ECAI 2022.',
    description='PyTorch implementation of ABLE.',
    epilog='end',
    add_help=True
)

parser.add_argument('--dataset', default='cifar10', \
                    help='dataset name', type=str, \
                    choices=['mnist', 'fmnist', 'kmnist', 'cifar10'])

parser.add_argument('--data-dir', default='./data', type=str)

parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')

parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')

parser.add_argument('--pmodel_path', default='./pmodel/cifar10.pt', type=str,
                    help='pretrained model path for generating instance dependent partial labels')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', \
                    choices=['resnet18'],
                    help='network architecture (only resnet18 used)')

parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')

parser.add_argument('--temperature', type=float, default=0.07,
                    help='temperature for loss function')

parser.add_argument('--loss_weight', default=1.0, type=float,
                    help='contrastive loss weight')

parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str, \
                    help='which gpu(s) can be used for distributed training')

parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--path', default='./image_output/image_1_label8.png', type=str,
                    help='the path to picture. ')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_VISIBLE_DEVICES


path = args.path 

cifar10_classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

def top5(value):
    value = value[0].numpy()
    value = value.astype(float)
    value = (value - np.min(value)) / (np.max(value) - np.min(value))
    combined = sorted(zip(value, cifar10_classes), key=lambda x: x[0], reverse=True)
    top5 = {}
    for i in range(5):
        top5[combined[i][1]] = combined[i][0]
    print(top5)
    return top5


def main(img_path=''):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
    
    print("=> creating ABLE_model '{}'\n".format(args.arch))

    model = ABLE(args=args, base_encoder=ABLENet)
    model.load_state_dict(torch.load('pretrain/ckpt60.pth'))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    output,flag = test_one_image(args, img_path, model) #这个就是置信区间
    print(output,flag)
    if flag == False:
        return {'error':0}
    #print('output: {}.\n'.format(output))

    # 将Tensor从CUDA设备移动到CPU
    cpu_tensor = output.to('cpu')
    print(cpu_tensor)
    max_index = torch.argmax(cpu_tensor, dim=1)
    number = max_index.item()
    print(cifar10_classes[number])
    return top5(cpu_tensor)


def test_one_image(args, path, model):    
    with torch.no_grad():      
        model.eval()   
        top1_acc = AverageMeter("Top1")
        img = Image.open(path)
        #img = img.resize((1024,1024))
        img = np.array(img).transpose(2, 0, 1)  # 表示C x H x W
        img = img.astype(np.float32)
        img_size = img.shape
        if img_size[0]!=3 or img_size[1]>1024 or img_size[2]>1024:
            return None, False
        img = np.expand_dims(img, axis=0)
        image = torch.tensor(img / 255)  # 神经网络里的张量数值一般在0-1之间归一化处理，所以除以255
        image= image.cuda()
        print("image.size", image.shape)
        output = model(args=args, img_w=image, is_eval=True)


    return output, True



if __name__ == '__main__':
    main()