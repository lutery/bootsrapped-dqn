# 参考连接：https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0?utm_source=chatgpt.com

# extending on code from
# https://github.com/58402140/Fruit
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import copy
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Size of game grid e.g. 10 -> 10x10
GRID_SIZE = 10
# How often to check and evaluate
EVALUATE_EVERY = 10
# Save images of policy at each evaluation if True, otherwise only at the end if False
SAVE_IMAGES = False
# How often to print statistics
PRINT_EVERY = 1

# Whether to use double DQN or regular DQN
USE_DOUBLE_DQN = True
# TARGET_UPDATE how often to use replica target
TARGET_UPDATE = 10
# Number of evaluation episodes to run
N_EVALUATIONS = 100
# Number of heads for ensemble (1 falls back to DQN)
N_ENSEMBLE = 5
# Probability of experience to go to each head
BERNOULLI_P = 1.
# Weight for randomized prior, 0. disables
PRIOR_SCALE = 1.
# Number of episodes to run
N_EPOCHS = 1000
# Batch size to use for learning
BATCH_SIZE = 128
# Buffer size for experience replay
BUFFER_SIZE = 1000
# Epsilon greedy exploration ~prob of random action, 0. disables
EPSILON = .0
# Gamma weight in Q update
GAMMA = .8
# Gradient clipping setting
CLIP_GRAD = 1
# Learning rate for Adam
ADAM_LEARNING_RATE = 1E-3

random_state = np.random.RandomState(11)

def seed_everything(seed=1234):
    #random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(22)

def save_img(epoch):
    if 'images_{}'.format(epoch) not in os.listdir('.'):
        os.mkdir('images_{}'.format(epoch))
    frame = 0
    while True:
        screen, reward = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.title("reward: {}".format(reward))
        plt.savefig('images_{}/{}.png'.format(epoch, frame))
        frame += 1

def episode():
    """
    Coroutine of episode.

    Action has to be explicitly sent to this coroutine.
    actions are 0, 1, 2 for left, don't-move, and right
    """
    x, y, z = (
        random_state.randint(0, GRID_SIZE),  # X of fruit
        0,  # Y of dot
        random_state.randint(1, GRID_SIZE - 1)  # X of basket
    )
    while True:
        X = np.zeros((GRID_SIZE, GRID_SIZE))  # Reset grid
        X = X.astype("float32")
        X[y, x] = 1.  # Draw fruit
        bar = range(z - 1, z + 2)
        X[-1, bar] = 1.  # Draw basket

        # End of game is known when fruit is at penultimate line of grid.
        # End represents either a win or a loss
        end = int(y >= GRID_SIZE - 2)

        reward = 0
        # can add this for dense rewards
        #if x in bar:
        #   reward = 1
        if end and x not in bar:
           reward = -1
        if end and x in bar:
           reward = 1

        action = yield X[None], reward #end
        if end:
            break

        # translate actions
        # 0 is left
        # 1 is same (0)
        # 2 is right
        action = action - 1

        z = min(max(z + action, 1), GRID_SIZE - 2)
        y += 1


def experience_replay(batch_size, max_size):
    """
    Coroutine of experience replay.

    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        inds = np.arange(len(memory))
        experience = yield [memory[i] for i in random_state.choice(inds, size=batch_size, replace=True)] if batch_size <= len(memory) else None
        # send None to just get random experiences, without changing buffer
        if experience is not None:
            memory.append(experience)
            if len(memory) > max_size:
                memory.pop(0)


class CoreNet(nn.Module):
    def __init__(self):
        super(CoreNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(-1, 10 * 10 * 16)
        return x


class HeadNet(nn.Module):
    def __init__(self):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(10 * 10 * 16, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet()
        self.net_list = nn.ModuleList([HeadNet() for k in range(n_ensemble)])
    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        return self.net_list[k](self.core_net(x))


class NetWithPrior(nn.Module):
    def __init__(self, net, prior, prior_scale=1.):
        super(NetWithPrior, self).__init__()
        self.net = net
        self.prior_scale = prior_scale
        if self.prior_scale > 0.:
            self.prior = prior

    def forward(self, x, k):
        '''
        x: 观察
        k： 选择的head索引，如果为None，则返回所有head的输出
        '''
        if hasattr(self.net, "net_list"): # 确保有多头
            if k is not None:
                if self.prior_scale > 0.:
                    # 使用指定的head进行前向传播，并加上先验动作网络的Q值输出 todo 作用
                    return self.net(x, k) + self.prior_scale * self.prior(x, k).detach()
                else:
                    # 使用指定的head进行前向传播
                    return self.net(x, k)
            else:
                core_cache = self.net._core(x) # 获取特征
                net_heads = self.net._heads(core_cache) # 获取所有动作Q值 head的输出
                if self.prior_scale <= 0.: # 没有先验网络
                    return net_heads # 直接返回所有head的输出
                else:
                    prior_core_cache = self.prior._core(x) # 先验网络的特征
                    prior_heads = self.prior._heads(prior_core_cache) # 先验网络的所有head输出
                    return [n + self.prior_scale * p.detach() for n, p in zip(net_heads, prior_heads)] # 返回加上先验网络的所有head输出
        else:
            raise ValueError("Only works with a net_list model")

prior_net = EnsembleNet(N_ENSEMBLE)
policy_net = EnsembleNet(N_ENSEMBLE)
policy_net = NetWithPrior(policy_net, prior_net, PRIOR_SCALE)

target_net = EnsembleNet(N_ENSEMBLE)
target_net = NetWithPrior(target_net, prior_net, PRIOR_SCALE)

opt = optim.Adam(policy_net.parameters(), lr=ADAM_LEARNING_RATE)

# change minibatch setup to use masking...
exp_replay = experience_replay(BATCH_SIZE, N_ENSEMBLE * BUFFER_SIZE) # todo
next(exp_replay) # Start experience-replay coroutines

# Stores the total rewards from each evaluation, per head over all epochs
accumulation_rewards = []
overall_time = 0.
for i in range(N_EPOCHS):
    start = time.time()
    ep = episode() # 创造游戏环境
    S, won = next(ep)  # Start coroutine of single episode 相当于reset，得到初始的状态S和奖励won
    epoch_losses = [0. for k in range(N_ENSEMBLE)] # 记录每个头的损失
    epoch_steps = [1. for k in range(N_ENSEMBLE)] # 一轮游戏每个头的步数
    # 随机选择一个头作为本轮游戏的激活头
    heads = list(range(N_ENSEMBLE)) 
    random_state.shuffle(heads)
    active_head = heads[0] # choose a head to use for this episode 这个激活头用于预测选择执行的动作
    try:
        policy_net.train()
        while True:
            if random_state.rand() < EPSILON:
                # 随机选择动作
                action = random_state.randint(0, 3)
            else: # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                with torch.no_grad():
                    # 选择一个激活头预测执行的动作
                    action = np.argmax(policy_net(torch.Tensor(S[None]),active_head).detach().data.numpy(), axis=-1)[0]

            # 发送执行的动作，看来python的yield可以双向传递数据
            S_prime, won = ep.send(action)
            ongoing_flag = 1. # 类似游戏结束标识
            exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE) # 计算head的经验掩码
            experience = (S, action, won, S_prime, ongoing_flag, exp_mask) # 记录经验
            S = S_prime # 更新状态
            batch = exp_replay.send(experience) # 利用yield传递经验，并获取一个批次的经验
            if batch:
                inputs = []
                actions = []
                rewards = []
                nexts = []
                ongoing_flags = []
                masks = []
                for b_i in batch:
                    s, a, r, s_prime, ongoing_flag, mask = b_i
                    rewards.append(r)
                    inputs.append(s)
                    actions.append(a)
                    nexts.append(s_prime)
                    ongoing_flags.append(ongoing_flag)
                    masks.append(mask)
                mask = torch.Tensor(np.array(masks))

                # precalculate the core Q values for every head
                all_target_next_Qs = [n.detach() for n in target_net(torch.Tensor(nexts), None)] # 计算下一个状态的动作Q值，不传递梯度
                all_Qs = policy_net(torch.Tensor(inputs), None)  # 计算当前状态的动作Q值，传递梯度
                if USE_DOUBLE_DQN:
                   all_policy_next_Qs = [n.detach() for n in policy_net(torch.Tensor(nexts), None)] # 这里利用policy_net计算下一个状态的动作Q值，用于Double DQN？不传递梯度
                # set grads to 0 before iterating heads
                opt.zero_grad()

                for k in range(N_ENSEMBLE): # 遍历每一个头
                    if USE_DOUBLE_DQN:
                        policy_next_Qs = all_policy_next_Qs[k] # 获取对应头的下一个状态的动作Q值 当前网络
                        next_Qs = all_target_next_Qs[k] # 获取对应头的下一个状态的动作Q值 目标网络
                        policy_actions = policy_next_Qs.max(1)[1][:, None] # 选择当前头最大的动作的索引
                        next_max_Qs = next_Qs.gather(1, policy_actions) # 根据选择的动作索引获取目标网络对应的动作Q值
                        next_max_Qs = next_max_Qs.squeeze()
                    else:
                        next_Qs = all_target_next_Qs[k] # 获取当前头的下一个状态的动作Q值
                        next_max_Qs = next_Qs.max(1)[0] # 获取当前头最大的动作Q值
                        next_max_Qs = next_max_Qs.squeeze()

                    # mask based on if it is end of episode or not
                    next_max_Qs = torch.Tensor(ongoing_flags) * next_max_Qs # 如果是结束状态，则将下一个状态的最大Q值设为0
                    target_Qs = torch.Tensor(np.array(rewards).astype("float32")) + GAMMA * next_max_Qs # 计算目标Q值

                    # get current step predictions
                    Qs = all_Qs[k] # 获取对应头的当前状态的动作Q值
                    Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32")))
                    Qs = Qs.squeeze()

                    # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
                    # need to do updates on each head based on experience mask
                    full_loss = (Qs - target_Qs) ** 2 # 计算均方误差损失，并保持维度
                    full_loss = mask[:, k] * full_loss # 根据经验掩码选择性地更新损失，将掩码为0的损失设为0，不传递梯度不训练
                    loss = torch.mean(full_loss) # 计算平均损失
                    #loss = F.smooth_l1_loss(Qs, target_Qs[:, None])

                    loss.backward(retain_graph=True)
                    for param in policy_net.parameters():
                        if param.grad is not None:
                            # Multiply grads by 1 / K
                            param.grad.data *= 1. / N_ENSEMBLE # 平均每个头的梯度，这个操作很重要，等同于计算每个头的平均损失，因为计算每个head的loss时，会计算梯度，而特征提取网络部分的梯度会不断的重复累积，所以需要除以N_ENSEMBLE
                            # todo 这里存在问题，head头的梯度不会重复计算，所以这里平均会导致head头的梯度变小 查看md
                    epoch_losses[k] += loss.detach().cpu().numpy() # 记录每个头的损失
                    epoch_steps[k] += 1. # 记录每个头步数
                # After iterating all heads, do the update step
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), CLIP_GRAD) # 梯度裁剪
                opt.step() # 优化器更新参数
    except StopIteration:
        # add the end of episode experience 如果yield不再返回数据，则表示游戏结束，会抛出异常
        ongoing_flag = 0. # 将结束标识设为0
        # just put in S, since it will get masked anyways 并将最后的经验加入经验回放
        exp_mask = random_state.binomial(1, BERNOULLI_P, N_ENSEMBLE)
        experience = (S, action, won, S, ongoing_flag, exp_mask)
        exp_replay.send(experience)

    stop = time.time()
    overall_time += stop - start # 一轮游戏的时间

    if TARGET_UPDATE > 0 and i % TARGET_UPDATE == 0:
        # 将策略网络的参数复制到目标网络
        print("Updating target network at {}".format(i))
        target_net.load_state_dict(policy_net.state_dict())

    if i % PRINT_EVERY == 0:
        # 打印训练的统计信息
        print("Epoch {}, head {}, loss: {}".format(i + 1, active_head, [epoch_losses[k] / float(epoch_steps[k]) for k in range(N_ENSEMBLE)]))

    if i % EVALUATE_EVERY == 0 or i == (N_EPOCHS - 1):
        # 评估当前策略网络的表现
        if i == (N_EPOCHS - 1):
            # 最后一次训练
            # save images at the end for sure
            SAVE_IMAGES = True # 保存图片
            ORIG_N_EVALUATIONS = N_EVALUATIONS # 验证的步数
            N_EVALUATIONS = 5
        if SAVE_IMAGES:
            img_saver = save_img(i) # 将游戏的图片保起来
            next(img_saver) #返回一个yield，用于后续评估时保存环境观察图片
        evaluation_rewards = []
        for _ in range(N_EVALUATIONS):
            g = episode()
            S, reward = next(g)
            reward_trace = [reward]
            if SAVE_IMAGES:
                img_saver.send((S, reward))
            try:
                policy_net.eval()
                while True:
                    acts = [np.argmax(q.data.numpy(), axis=-1)[0] for q in policy_net(torch.Tensor(S[None]), None)] # 获取每一个头预测的最大Q值的动作
                    act_counts = Counter(acts) # 并统计每个动作被选择的次数
                    max_count = max(act_counts.values()) # 获取最大选择次数
                    top_actions = [a for a in act_counts.keys() if act_counts[a] == max_count] # 选择被最多头选择的动作
                    # break action ties with random choice
                    random_state.shuffle(top_actions) # 如果多个动作被相同次数选择，则随机选择一个
                    act = top_actions[0]
                    S, reward = g.send(act) # 执行动作
                    reward_trace.append(reward) # 记录奖励
                    if SAVE_IMAGES:
                        img_saver.send((S, reward)) # 保存
            except StopIteration:
                # sum should be either -1 or +1
                evaluation_rewards.append(np.sum(reward_trace)) # 评估一轮游戏的总奖励
        accumulation_rewards.append(np.mean(evaluation_rewards)) # 记录评估的平均奖励
        print("Evaluation reward {}".format(accumulation_rewards[-1]))
        if SAVE_IMAGES:
            img_saver.close() # 关闭图片保存协程

    if i == (N_EPOCHS - 1): # 最后一次训练，绘制奖励曲线
        plt.figure()
        trace = np.array(accumulation_rewards)
        xs = np.array([int(n * EVALUATE_EVERY) for n in range(N_EPOCHS // EVALUATE_EVERY + 1)])
        plt.plot(xs, trace, label="Reward")
        plt.legend()
        plt.ylabel("Average Evaluation Reward ({})".format(ORIG_N_EVALUATIONS))

        model = "Double DQN" if USE_DOUBLE_DQN else "DQN"
        # 下面就是输出一些设置参数
        if N_ENSEMBLE > 1:
            model = "Bootstrap " + model
        if PRIOR_SCALE > 0.:
            model = model + " with randomized prior {}".format(PRIOR_SCALE)
        footnote_text = "Episodes\n"
        footnote_text += "\n"
        footnote_text += "\n"
        footnote_text += "Settings:\n"
        footnote_text += "{}\n".format(model)
        footnote_text += "Number of heads {}\n".format(N_ENSEMBLE)
        footnote_text += "Epsilon-greedy {}\n".format(EPSILON)
        if N_ENSEMBLE > 1:
            footnote_text += "Sharing mask probability {}\n".format(BERNOULLI_P)
        footnote_text += "Gamma decay {}\n".format(GAMMA)
        footnote_text += "Grad clip {}\n".format(CLIP_GRAD)
        footnote_text += "Adam, learning rate {}\n".format(ADAM_LEARNING_RATE)
        footnote_text += "Batch size {}\n".format(BATCH_SIZE)
        footnote_text += "Experience replay buffer size {}\n".format(BUFFER_SIZE)
        footnote_text += "Training time {}\n".format(overall_time)
        plt.xlabel(footnote_text)
        plt.tight_layout()
        plt.savefig("reward_traces.png")
