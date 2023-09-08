"""
Models and transforms.

"""


import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate


#####################################################################################

class driftmodel(nn.Module): # create the model
    """
    Logistic model with drift
    """
    def __init__(self, input_size, pdrop=0): # modify 15 to 13 based on the number of features
        """
        input_size : number of features
        pdrop : dropout probability
        """
        super(driftmodel, self).__init__()
        # Settings
        self.input_size = input_size
        self.pdrop = pdrop
        # Layers
        self.fc = nn.Linear(input_size,1) # 全连接层，将输入x转换成1维张量。
        self.rfc = nn.Linear(1,1, bias=False) # 全连接层。
        self.sigmoid = nn.Sigmoid() # Sigmoid函数，将y转换成0-1之间的值。
        self.drop = nn.Dropout(p=pdrop) # Dropout层，防止过拟合。
        self.relu = nn.ReLU()
    
    # forward函数实现了模型的前向传播过程，接收时刻t下的输入x和上一个时刻t-1下的输出y，返回当前时刻t下的输出y。

    def forward(self, t, x, y):
        """
        t : time step
        x : input at time t
        y : output at time t-1
        """
        act = self.rfc(y-.5) + self.fc(self.drop(x))
        y = self.sigmoid(act)
        return y



#####################################################################################

class TimeWarping(object):
    """
    Warping transform over the time variable
    """
    def __init__(self, deltarange, randomdelta=True):
        """
        deltarange: exponent for time warping via t_new = T*(t_old/T)**delta
        randomdelta: if True, randomly choose exponent in [1/deltarange, deltarange]
        """
        super(TimeWarping, self).__init__()
        self.deltarange = deltarange
        self.randomdelta = randomdelta

    def __call__(self, x):
        if self.deltarange!=1:
            if self.deltarange < 1:
                self.deltarange = 1/self.deltarange
            delta = (self.deltarange - 1/self.deltarange)*np.random.random_sample() \
                    + 1/self.deltarange if self.randomdelta else self.deltarange
            timebins = x.shape[-1] # tensor is (features x timebins)
            newtimes = (timebins-1)*(np.arange(timebins)/(timebins-1))**delta
            f = interpolate.interp1d(np.arange(timebins), x)
            x = torch.tensor(f(newtimes))

        return x

#####################################################################################

def compute_pvalues(seq, dmean=None, optsided='two', direction=None):
    """
    seq = sequence of outputs (n_perms+1 x n_out)
          seq[0,:] = original values
          seq[1:,:] = values from permuted input
    dmean = distribution mean
    optsided = 'one' for one-sided, 'two' for two-sided
    direction = 1 or -1, for one-sided case (inferred from data if not specified)
    """

    seq = np.array(seq)
    if len(seq.shape)==1:
        seq = np.expand_dims(seq,1)
    if dmean==None:
        dmean = np.mean(seq[1:,:], axis=0) # permuation; mean 1~nperm
    seq -= dmean # subtract the permutation mean
    nPerms = seq.shape[0] - 1 
    nOut = seq.shape[1]
    
    # 对于单侧检验，首先初始化输出的p值数组，然后对于每个输出，根据方向（大于还是小于），计算排列数据中大于等于或小于等于原始值的数量，并将其除以排列数，作为p值。
    
    if optsided=='one':
        pvalues = np.zeros(nOut)
        for iOut in range(nOut):
            one_sided_direction = np.sign(seq[0,iOut]) if direction==None else direction
            if one_sided_direction > 0:
                pvalues[iOut] = np.sum(seq[1:,iOut] >= seq[0,iOut])/nPerms
            else:
                pvalues[iOut] = np.sum(seq[1:,iOut] <= seq[0,iOut])/nPerms

    # 对于双侧检验，对于每个输出，计算在排列数据中大于等于或小于等于原始值的数量，并将其除以排列数，作为p值。
            
    elif optsided=='two': # perm beta >= normal beta -> number/nPerms -> p values
        pvalues = np.sum(np.abs(seq[1:,:]) >= np.abs(seq[0,:]), 0)/nPerms
    else:
        print('Invalid optsided value (must be either \'one\' or \'two\')')

    return pvalues
