#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
# from utee import misc
from collections import OrderedDict
import torch.nn.init as init
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(1024,128)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(128,10)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['conv1.weight', 'conv1.bias'],['conv2.weight', 'conv2.bias'],['conv3.weight', 'conv3.bias'],['conv4.weight', 'conv4.bias'],['conv5.weight', 'conv5.bias'],['conv6.weight', 'conv6.bias'],['conv7.weight', 'conv7.bias'],['conv8.weight', 'conv8.bias'],['conv9.weight', 'conv9.bias'],['conv10.weight', 'conv10.bias'],['conv11.weight', 'conv11.bias'],['conv12.weight', 'conv12.bias'],['conv13.weight', 'conv13.bias'],
                            ['fc14.weight', 'fc14.bias'],
                            ['fc15.weight', 'fc15.bias'],
                            ['bn1.weight', 'bn1.bias'],
                            ['bn2.weight', 'bn2.bias'],['bn3.weight', 'bn3.bias'],['bn4.weight', 'bn4.bias'],['bn5.weight', 'bn5.bias'],
                            ['fc16.weight', 'fc16.bias']
                            ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1)

class lenet(torch.nn.Module):
    def __init__(self, args):
        super(lenet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=3),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class lenetMini(torch.nn.Module):
    def __init__(self, args):
        super(lenet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=3),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, 128),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(args=None):
    return ResNet(BasicBlock, [3, 3, 3])



def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float32')
    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
    return x / np.sum(x)



def LCN(image_tensor, gaussian, mid):
    filtered= gaussian(image_tensor)
    centered_image = image_tensor - filtered[:,:,mid:-mid,mid:-mid]
    sum_sqr_XX = gaussian(centered_image.pow(2))
    denom = sum_sqr_XX[:,:,mid:-mid,mid:-mid].sqrt()
    per_img_mean = denom.mean()
    divisor = denom.clone()
    divisor[per_img_mean > denom ] =per_img_mean
    divisor[divisor < 1e-4 ] = 1e-4
    new_image = centered_image / divisor
    return new_image




class STNet(nn.Module):
    def __init__(self, args):
        super(STNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, kernel_size=7 ,stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2 , ceil_mode=True)
        self.gfilter1 = torch.Tensor(gaussian_filter((1,200,9,9)) )
        self.gaussian1 = nn.Conv2d(in_channels=200, out_channels=200,
                            kernel_size=9  , padding= 8 , bias=False)
        self.gaussian1.weight.data = self.gfilter1
        self.gaussian1.weight.requires_grad = False
        self.conv2 = nn.Conv2d(200, 250, kernel_size=4 ,stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2 , ceil_mode=True)
        self.gfilter2 = torch.Tensor(gaussian_filter((1,250,9,9)) )
        self.gaussian2  = nn.Conv2d(in_channels=250, out_channels=250,
                            kernel_size=9  , padding= 8 , bias=False)
        self.gaussian2.weight.data = self.gfilter2
        self.gaussian2.weight.requires_grad = False
        self.conv3 = nn.Conv2d(250, 350, kernel_size=4 ,stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.gfilter3 = torch.Tensor(gaussian_filter((1,350,9,9)) )
        self.gaussian3  = nn.Conv2d(in_channels=350, out_channels=350,
                            kernel_size=9  , padding= 8 , bias=False)
        self.gaussian3.weight.data = self.gfilter3
        self.gaussian3.weight.requires_grad = False
        self.FC1 = nn.Linear(12600, 400)
        self.FC2 = nn.Linear(400, 43)
        
        #Spatial Attention Model, Spatial Transformers Layers
        self.st1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2 , ceil_mode=True),
            nn.Conv2d(3, 250, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=True),
            nn.Conv2d(250, 250, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=True)
        )
        self.FC1_ = nn.Sequential(
            nn.Linear(9000, 250),
            nn.ReLU(True),
            nn.Linear( 250 , 6 )
        )
        self.st2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2 , ceil_mode=False),
            nn.Conv2d(200, 150, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=False),
            nn.Conv2d(150, 200, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=False)
        )
        self.FC2_ =  nn.Sequential(
            nn.Linear(800, 300),
            nn.ReLU(True),
            nn.Linear( 300 , 6 )
        )
        self.st3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2 , ceil_mode=False),
            nn.Conv2d(250, 150, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=False),
            nn.Conv2d(150, 200, kernel_size=5 ,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2 , ceil_mode=False)
        )
        self.FC3_ =  nn.Sequential(
            nn.Linear(200, 300),
            nn.ReLU(True),
            nn.Linear( 300 , 6 )
        )
        self.FC1_[2].weight.data.zero_()
        self.FC1_[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.FC2_[2].weight.data.zero_()
        self.FC2_[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.FC3_[2].weight.data.zero_()
        self.FC3_[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        #First Layer is the Spatial Transformer Layer
        #ST-1
        h1 = self.st1(x)
        h1 = h1.view(-1, 9000)
        h1 = self.FC1_(h1)
        theta1 = h1.view(-1, 2, 3)
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        x = F.grid_sample(x, grid1, align_corners=False)
        
        #Convolution, Relu and Maxpool , SET #1
        x = F.relu(self.conv1(x))
        x =  self.maxpool1(x)
        
        #Paper Says to apply LCN here, but LCN Layer Before Convolution Worked for me better 
        #ST-2
        h2 = self.st2(x)
        h2=h2.view(-1,800)
        h2 = self.FC2_(h2)
        theta2 = h2.view(-1, 2, 3)
        grid2 = F.affine_grid(theta2, x.size(), align_corners=False)
        x = F.grid_sample(x, grid2, align_corners=False)
        
        #LCN Layer : Based on paper implemntation from the github and Yann Lecun Paper 2009
        mid1 = int(np.floor(self.gfilter1.shape[2] / 2.))
        x = LCN(x , self.gaussian1, mid1)
        
        #Convolution, Relu and Maxpool , SET #2
        x = F.relu(self.conv2(x))
        x=  self.maxpool2(x)
        
        #ST-2
        h3 = self.st3(x)
        h3 = h3.view(-1, 200)
        h3 = self.FC3_(h3)
        theta3 = h3.view(-1, 2, 3)
        grid3 = F.affine_grid(theta3, x.size(), align_corners=False)
        x = F.grid_sample(x, grid3, align_corners=False)
        
        #LCN Layer : 2
        mid2 = int(np.floor(self.gfilter2.shape[2] / 2.))
        x = LCN(x , self.gaussian2, mid2)

        #Convolution, Relu and Maxpool , SET #3
        x = F.relu(self.conv3(x))
        x=  self.maxpool3(x)

        #LCN Layer : 3
        mid3 = int(np.floor(self.gfilter3.shape[2] / 2.))
        x = LCN(x , self.gaussian3, mid3)
        
        #Dimensions in accordance to paper
        y = x.view(-1, 12600)
        y = F.relu(self.FC1(y))
        y = self.FC2(y)
        return F.log_softmax(y, dim=1)
        
