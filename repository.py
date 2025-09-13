import torch
from collections import deque
from collections import namedtuple
import numpy as np
import random
from skimage.transform import rescale
from skimage.transform import resize
import copy

class memoryDataset(object):
    def __init__(self, maxlen, n_ensemble=1, bernoulli_prob=0.9):
        '''
        maxlen: int, maximum length of memory 缓冲区的长度
        n_ensemble: int, number of ensemble heads
        bernoulli_prob: float, probability of each head to be trained on each transition
         0 < bernoulli_prob <= 1
         if bernoulli_prob = 1, then all heads are trained on all transitions
         if bernoulli_prob = 0.5, then each head is trained on half of the transitions
         if bernoulli_prob = 0, then no head is trained on any transition
        '''
        self.memory = deque(maxlen=maxlen)
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob

        ## if ensemble is 0 then no need to apply mask
        # todo 为啥？
        if n_ensemble==1:
            self.bernoulli_prob = 1

        # 声明一个命名元组，用于存储状态转换信息
        self.subset = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'life', 'terminal', 'mask'))


    def push(self, state, action, next_state, reward, done, life, terminal):
        '''
        state: np.array, current state 当前状态
        action: int, action taken 执行动作
        next_state: np.array, next state after taking action 执行动作后的下一个状态
        reward: float, reward received after taking action 执行动作后获得的奖励
        done: boolean, whether the episode is done 是否结束
        life: int, number of lives left 剩余生命数
        terminal: boolean, whether the episode is done due to losing a life 是否因为失去生命而结束
        '''

        state = np.array(state)
        action = np.array([action])
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array([done])
        life = np.array([life])
        terminal = np.array([terminal])
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble) # todo 作用？看起来是生成一个伯努利分布的掩码，作用是啥？

        # 将收集的状态信息存储到缓冲区中
        self.memory.append(self.subset(state, action, next_state, reward, done, life, terminal, mask))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        # 从缓冲区中随机采样一个批次的状态转换信息，不是连续的
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        # 将采样的batch，重新解包重组，将state、action等分别堆叠在一起
        batch = self.subset(*zip(*batch)) # todo 学习

        # 将numpy array转换为torch tensor重新打包为命名元组
        state = torch.tensor(np.stack(batch.state), dtype=torch.float)
        action = torch.tensor(np.stack(batch.action), dtype=torch.long)
        reward = torch.tensor(np.stack(batch.reward), dtype=torch.float)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float)

        done = torch.tensor(np.stack(batch.done), dtype=torch.long)
        ##Life : 0,1,2,3,4,5
        life = torch.tensor(np.stack(batch.life), dtype=torch.float)
        terminal = torch.tensor(np.stack(batch.terminal), dtype=torch.long)
        mask = torch.tensor(np.stack(batch.mask), dtype=torch.float)
        batch = self.subset(state, action, next_state, reward, done, life, terminal, mask)

        return batch

class historyDataset(object):
    '''
    todo 作用
    对观察样本进行预处理（包含裁剪、缩放、图片多通道合并二值化、帧堆叠）
    '''
    def __init__(self, history_size, img, crop_flag=False):
        '''
        history_size: int, number of frames to stack 帧堆叠的长度
        img: np.array, initial observation image 输出图像的观察图片
        crop_flag: boolean, whether to crop the image (for breakout) 是否对图片进行裁剪，去除多余的无效区域
        '''
        self.history_size = history_size
        self.crop_flag = crop_flag

        state = self.convert_channel(img)
        self.height, self.width = state.shape

        temp = []
        for _ in range(history_size):
            temp.append(state)
        self.history = temp

    def convert_channel(self, img):
        # input type : |img| = (Height, Width, channel)
        # remove useless item

        if self.crop_flag:
            # 裁减无效的区域
            img = img[31:193, 8:152]

        #img = rescale(img, 1.0 / 2.0, anti_aliasing=False, multichannel=False)
        # 将图片转换为84x84
        img = resize(img, output_shape=(84, 84))

        # conver channel(3) -> channel(1)
        # 对每个像素的RGB值进行逻辑或操作，只要有一个通道非零，结果就为True
        # 这样可以将彩色图像转换为二值图像，保留了图像中的边缘和形状信息
        img = np.any(img, axis=2)
        # |img| = (Height, Width)  boolean
        return img

    def push(self, img):
        '''
        img： np.array, new observation image 新的额观察数据
        '''
        temp = self.history
        state = self.convert_channel(img)
        temp.append(state)
        self.history = temp[1:] # 将新的观察图像加入历史记录，并移除最旧的图像，貌似时帧堆叠

    def get_state(self):
        #return self.history
        return copy.deepcopy(self.history)
