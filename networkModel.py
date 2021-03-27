import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, input_size, n_unit=[16]*4):
        super(Predictor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, n_unit[0], bias=True)
        self.fc2 = nn.Linear(n_unit[0], n_unit[1], bias=True)
        self.fc3 = nn.Linear(n_unit[1], n_unit[2], bias=True)
        self.fc4 = nn.Linear(n_unit[2], n_unit[3], bias=True)
        self.fc5 = nn.Linear(n_unit[3], 1, bias=True)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        return x

class FullConnectGenerator(nn.Module):
    def __init__(self, n_unit=[16,32,32,32], input_size=7, output_size=1): # ネットワークで使う関数を定義する。
        super(FullConnectGenerator, self).__init__()

        self.fc1 = nn.Linear(input_size, n_unit[0], bias=True)
        self.fc2 = nn.Linear(n_unit[0], n_unit[1], bias=True)
        self.fc3 = nn.Linear(n_unit[1], n_unit[2], bias=True)
        self.fc4 = nn.Linear(n_unit[2], n_unit[3], bias=True)
        self.fc5 = nn.Linear(n_unit[3], output_size, bias=True)

            
    def forward(self, x):# ここでネットワークを構成する。入力はx。
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, n_unit=[64]*3):
        super(Discriminator, self).__init__()
        
        self.sigma = nn.Parameter(torch.tensor([1.0]))
        self.fc1 = nn.Linear(input_size, n_unit[0])
        self.fc2 = nn.Linear(n_unit[0], n_unit[1])        
        self.fc3 = nn.Linear(n_unit[1], n_unit[2])        
        self.fc4 = nn.Linear(n_unit[2], 1)        
        
    def forward(self, x, is_from_generator=True):
        if not is_from_generator:# 標準正規分布からのサンプリングが入力の場合は定数倍する
            x = self.sigma*x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class LinearGenerator(nn.Module):
    def __init__(self, input_size=7, is_bias=False): # ネットワークで使う関数を定義する。
        super(LinearGenerator, self).__init__()
        # 線形変換: y = Wx + b
        self.fc1 = nn.Linear(in_features=input_size, out_features=1, bias=is_bias)
            
    def forward(self, x):# ここでネットワークを構成する。入力はx。
        x = self.fc1(x)
        return x # 出力

class LinearPredictor(nn.Module):
    def __init__(self, input_size=7, is_bias=False): # ネットワークで使う関数を定義する。
        super(LinearPredictor, self).__init__()
        # 線形変換: y = Wx + b
        self.fc1 = nn.Linear(input_size, 1, bias=is_bias)

    def forward(self, x):# ここでネットワークを構成する。入力はx。
        # x = x.view(x.shape[0],-1) 
        x = self.fc1(x)
        return x # 出力