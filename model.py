from torch import nn
from utills import init_weights, normalized_columns_initializer
import torch
import numpy as np
import torch.nn.functional as F


class HeadNet(nn.Module):
    def __init__(self, reshape_size, n_actions=4):
        '''
        reshape_size: 特征提取层的输出维度
        n_actions: int, number of actions 动作的维度
        '''
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, h, w, num_channels):
        '''
        n_ensemble: int, number of ensemble heads todo 作用
        n_actions: int, number of actions
        h: int, height of input image
        w: int, width of input image
        num_channels: int, number of channels of input image
        '''
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(h=h, w=w, num_channels=num_channels)
        reshape_size = self.core_net.reshape_size # CoreNet输出的特征维度
        # todo
        self.net_list = nn.ModuleList([HeadNet(reshape_size=reshape_size, n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k=None):
        '''
        X: input tensor, shape (batch_size, num_channels, h, w) 观察图片张量
        k: int or None, index of the head to use for forward pass. If None, return outputs of all heads. 
        使用哪个头进行前向传播，如果为None，则返回所有头的输出
        '''
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads


class CoreNet(nn.Module):
    def __init__(self, h, w, num_channels=4):
        '''
        核心网络，卷积层，提取观察特征
        h: int, height of input image 图片的高度
        w: int, width of input image 图片的宽度
        num_channels: int, number of channels of input image 图片的通道数
        '''
        super(CoreNet, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # 计算经过三层卷积后的特征图大小
        # todo 修改为更加简洁的形式，比如通过构建一个临时的卷积网络来计算，得到最终的特征图大小
        # 然后就可以计算出reshape_size
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        self.reshape_size = convw * convh * 64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3 将张量展平
        x = x.view(-1, self.reshape_size)
        return x


