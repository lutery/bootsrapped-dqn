import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import logging
from model import EnsembleNet
from properties import build_parser, CONSOLE_LEVEL, LOG_FILE, LOGFILE_LEVEL
from repository import historyDataset, memoryDataset
import sys
import traceback
from PIL import Image
from collections import Counter


CHECKPOINT_NAME = 'pytorch_model.bin'
CONFIG_NAME = "training_args.bin"

## 모델 불러오기
def load_saved_model(model, path):
    checkpoint = torch.load(os.path.join(path, CHECKPOINT_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
    return model

## 모델 저장
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, CHECKPOINT_NAME))

def save_config(config, path):
    torch.save(config, os.path.join(path, CONFIG_NAME))

def load_saved_config(path):
    return torch.load(os.path.join(path, CONFIG_NAME))

def save_numpy(temp_data, path, name):
    temp_path = os.path.join(path, name)
    np.save(temp_path, temp_data)

def build_model(config):
    # 构建动作策略模型
    return EnsembleNet(n_ensemble=config.n_ensemble, n_actions=config.class_num, h=config.resize_unit[0], w=config.resize_unit[1],
                num_channels=config.history_size)


class DQNSolver():

    def __init__(self, config):
        self.device = config.device
        self.env = gym.make(config.env) # 创建训练环境和验证环境，没有额外的环境包装（比如帧堆叠、跳帧等）
        self.valid_env = gym.make(config.env)
        self.memory_size = config.memory_size 
        self.update_freq = config.update_freq # 训练更新模型的频率
        self.learn_start = config.learn_start # 模型正式训练起始步数，估计有一段是再收集数据
        self.history_size = config.history_size

        self.batch_size = config.batch_size
        self.ep = config.ep # epsilon-贪婪策略中的初始epsilon
        self.eps_end = config.eps_end # epsilon-贪婪策略中的最小epsilon
        self.eps_endt = config.eps_endt # epsilon-贪婪策略中的epsilon衰减步数，也即是多少步后epsilon衰减到最小值
        self.eps_start = self.learn_start

        self.lr = config.lr
        self.discount = config.discount

        self.agent_type = config.agent_type
        self.max_steps = config.max_steps # 训练的最大步数
        self.eval_freq = config.eval_freq
        self.eval_steps = config.eval_steps
        self.target_update = config.target_update
        self.max_eval_iter = config.max_eval_iter

        ##Breakout Setting
        if config.pretrained_dir is not None:
            # 测试时加载预训练模型和预训练配置
            pretrained_config = load_saved_config(config.pretrained_dir)
            config.n_ensemble = pretrained_config.n_ensemble
            config.class_num = pretrained_config.class_num
            config.resize_unit = pretrained_config.resize_unit

            policy_model = build_model(config)
            target_model = build_model(config)
            self.policy_model = load_saved_model(policy_model, config.pretrained_dir)
            self.target_model = load_saved_model(target_model, config.pretrained_dir)

        else:
            config.resize_unit = (84, 84) # 训练的观察size=84
            config.class_num = self.env.action_space.n
            self.policy_model = build_model(config) # 构建动作策略模型
            self.target_model = build_model(config) # 构建目标动作策略模型

        self.resize_unit = config.resize_unit
        self.class_num = config.class_num # 动作数量
        self.n_ensemble = config.n_ensemble # 动作预测的头数

        self.policy_model.to(config.device)
        self.target_model.to(config.device)

        # 构建动作策略模型优化器
        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=self.lr)

        ##INIT Memory SETTING
        # 构建训练数据经验回放池
        self.memory = memoryDataset(maxlen=config.memory_size, n_ensemble=config.n_ensemble,
                                    bernoulli_prob=config.bernoulli_prob)

        ##INIT LOGGER
        # 清理日志系统已有的处理器，防止重复记录日志
        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                # 遍历所有的处理器并移除它们
                logging.getLogger().removeHandler(handler)
        # 设置日志配置，包含日志文件的名字和日志级别
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL) ## set log config
        # 设置控制台日志处理器，指定日志级别，保证控制器也能够输出日志
        console = logging.StreamHandler() # console out
        console.setLevel(CONSOLE_LEVEL) # set log level
        logging.getLogger().addHandler(console)

        ##save options
        # 构建输出目录
        self.out_dir = config.out_dir
        if not os.path.isdir(config.out_dir):
            os.mkdir(config.out_dir)

        self.test_score_memory = [] # 存储每次评估的得分
        self.test_length_memory = [] # 存储每次评估的步数
        self.train_score_memory = [] # 记录每次评估时，最近10个回合的平均得分
        self.train_length_memory = [] # 记录每次评估时，最近10个回合的平均步数

        ##중간시작
        self.start_steps = config.start_steps # 训练的起始步数
        self.learn_start = self.learn_start + self.start_steps
        self.eval_steps = self.eval_steps + self.start_steps

        self.config = config
        # 保存配置，也就是保存本次训练的参数，以便后续加载模型/训练模式时使用
        save_config(config, self.out_dir)

        # todo 作用
        self.refer_img = config.refer_img
        if self.refer_img is not None:
            assert os.path.isdir(self.refer_img), 'there is no reference image folder'

        # todo 作用
        self.crop_flag = False
        if 'breakout' in config.env.lower():
            self.crop_flag = True




    def choose_action(self, history, header_number:int=None, epsilon=None):
        '''
        history： historyDataset object, 观察样本处理对象
        header_number: int, number of ensemble head, 选择哪个头
        epsilon: float, epsilon for epsilon-greedy action selection, epsilon-贪婪策略中的epsilon
        '''
        if epsilon is not None:
            # 如果开启了epsilon-贪婪策略
            if np.random.random() <= epsilon:
                # 随机选中额一个动作
                return self.env.action_space.sample()
            else:
                # 如果没有随机选中动作，则使用目标模型选择动作
                with torch.no_grad():
                    state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to(self.device)
                    if header_number is not None:
                        # 如果指定了头编号，则使用指定的头进行动作选择
                        action = self.target_model(state, header_number).cpu()
                        return int(action.max(1).indices.numpy())
                    else:
                        # vote
                        # 没有指定头编号，则使用所有头进行投票选择动作
                        # 也就是每个头选择一个动作，然后选择动作Q值最大出现频率最高的动作
                        actions = self.target_model(state)
                        actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                        actions = Counter(actions)
                        action = actions.most_common(1)[0][0]
                        return action
        else:
            # 如果没有传递epsilon参数，则直接使用训练的动作策略模型选择动作
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to(self.device)
                if header_number is not None:
                    action = self.policy_model(state, header_number).cpu()
                    return int(action.max(1).indices.numpy())
                else:
                    # vote
                    actions = self.policy_model(state)
                    actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                    actions = Counter(actions)
                    action = actions.most_common(1)[0][0]
                    return action



    def get_epsilon(self, t):
        '''
        t: int, current time step 当前时间步
        根据步数t计算epsilon的值
        '''
        # (self.ep - self.eps_end)： 代表epsilon的衰减范围
        # max(0, t - self.eps_start)： 代表从起始步数开始计算的当前时间步数
        # (self.eps_endt - max(0, t - self.eps_start)): 代表剩余的衰减步数
        # (self.eps_endt - max(0, t - self.eps_start))/self.eps_endt : 代表剩余衰减步数占总衰减步数的比例
        # (self.ep - self.eps_end) * (self.eps_endt - max(0, t - self.eps_start)) / self.eps_endt : 代表当前时间步下，epsilon的真实值
        epsilon =  self.eps_end + max(0, (self.ep - self.eps_end)*(self.eps_endt - max(0, t - self.eps_start)) /self.eps_endt )
        return epsilon

    def replay(self, batch_size):
        '''
        batch_size: int, number of samples in a batch 批次大小
        '''
        self.optimizer.zero_grad()

        # 随机采样不连续的batch_size个样本
        batch = self.memory.sample(batch_size)

        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        next_state = batch.next_state.to(self.device)
        reward = batch.reward
        reward = reward.type(torch.bool).type(torch.float).to(self.device)

        done = batch.done.to(self.device)
        life = batch.life.to(self.device)
        terminal = batch.terminal.to(self.device)
        mask = batch.mask.to(self.device)

        with torch.no_grad():
            # 利用下一个状态计算下一个状态的动作值，这里可以理解为Q值，这里可以看成是计算每一个动作的Q值（有多少维度就有多少个Q值）
            next_state_action_values = self.policy_model(next_state)
        # 计算当前状态的动作值
        state_action_values = self.policy_model(state)

        total_loss = []
        for head_num in range(self.n_ensemble): # 遍历每一个动作预测头
            # 获取所有训练样本中指定头head_num的样本掩码之和
            total_used = torch.sum(mask[:, head_num])
            if total_used > 0.0: # 如果样本中有一个样本的mask是1， 则训练当前头
                # next_state_action_values[head_num]获取对应头的下一个状态动作值，包含所有的样本状态
                # torch.max(next_state_action_values[head_num], dim=1).values.view(-1, 1)获取对应头的下一个状态动作值的最大值，包含所有的样本状态
                next_state_value = torch.max(next_state_action_values[head_num], dim=1).values.view(-1, 1)
                reward = reward.view(-1, 1)
                # reward + (self.discount * next_state_value), 计算目标Q值
                # reward, 计算即时奖励
                # gather(1, terminal)，在维度1上根据terminal索引选择对应的值
                # 索引含义：
                # [:, 0] -> reward + discount * next_state_value  (非终止状态的Q值计算)
                # [:, 1] -> reward                                (终止状态的Q值计算)
                target_state_value = torch.stack([reward + (self.discount * next_state_value), reward], dim=1).squeeze().gather(1, terminal)
                # 当前状态的动作Q值根据模型基于现在的状态预测的动作值结合实际执行的动作得到
                state_action_value = state_action_values[head_num].gather(1, action)
                # 当前状态模型预测的Q值需要和目标Q值进行对齐得到loss
                # reduction='none'表示不进行任何归约操作，返回与输入张量相同形状的张量，这样会保持每个样本的loss
                # 方便后续只计算掩码为1的loss
                loss = F.smooth_l1_loss(state_action_value, target_state_value, reduction='none')
                # 只计算掩码为1的样本的loss，其他样本的loss置为0
                loss = mask[:, head_num] * loss
                # 计算当前头的平均loss，除以掩码为1的样本数
                loss = torch.sum(loss / total_used)
                total_loss.append(loss)

        if len(total_loss) > 0:
            # 计算所有头的平均loss
            # 然后开始反向传播和优化
            total_loss = sum(total_loss)/self.n_ensemble # 之前的loss计算，其中的梯度计算时 特征提取网络会重复累积，所以需要除以n_ensemble，但是动作预测头不需要除以n_ensemble，所以需要调整 todo 对比调整
            total_loss.backward()
            self.optimizer.step()


    def valid_run(self):
        '''
        评估模型
        '''

        state = self.valid_env.reset()
        # 构建初始的帧样本数据
        valid_history = historyDataset(self.history_size, state, self.crop_flag)
        score = 0 # 评估得分
        count = 0 # 评估步数
        terminal = True
        done = False
        last_life = 0

        ## put valid time limits to make it fast evaluation
        while not done and count < self.max_eval_iter:
            # 选择最大Q值出现频率最高的动作
            action = self.choose_action(valid_history)
            if terminal: ## There is error when it is just started. So do action = 1 at first
               # 游戏刚开始或者者生命值减少时，terminal会被置为True，则需要执行动作1
               action = 1
            # 执行动作
            next_state, reward, done, life = self.valid_env.step(action)
            valid_history.push(next_state)
            # 记录分数
            score += reward
            life = life['ale.lives']
            count = count + 1 # 更新步数

            ## Terminal options
            # 计算是否丢失了一条生命
            if life < last_life:
                terminal = True
            else:
                terminal = False
            last_life = life

        return score, count

    def render_policy_net(self):
        '''
        测试时并渲染模型的动作选择过程
        '''

        def get_concat_h(im1, im2):
            '''
            水平拼接图片
            '''
            baseheight = im1.size[1]
            # 计算im2的宽度，使得im2的高度和im1一致
            wpercent = (baseheight / float(im2.size[1]))
            wsize = int((float(im2.size[0]) * float(wpercent)))
            im2_modified = im2.resize((wsize, baseheight), Image.ANTIALIAS)

            # 创建一个新的图片，宽度是im1和im2_modified的宽度之和，高度是im1的高度
            dst = Image.new('RGB', (im1.width + im2_modified.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2_modified, (im1.width, 0))
            return dst

        def get_concat_v(im1, im2):
            if im1 is None:
                return im2

            dst = Image.new('RGB', (im1.width, im1.height + im2.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            return dst

        arrow_images = []
        # 看来当前代码主要是针对Breakout游戏的
        if self.refer_img is not None and 'breakout' in self.config.env.lower():
            # 这里加载参考图片，参考图片是一些箭头图片，用于渲染时显示动作选择
            arrow_files = [filename for filename in os.listdir(self.refer_img) if '.png' in filename]
            if len(arrow_files) >= self.class_num:
                arrow_images = [Image.open(os.path.join(self.refer_img, filename)) for filename in arrow_files]

        state = self.env.reset()
        history = historyDataset(self.history_size, state, self.crop_flag)
        score = 0 # 评估得分
        count = 0 # 评估步数
        raw_frames = []
        frames = [] # 存储评估过程中的每一帧图片
        done = False
        terminal = True
        last_life = 0
        while not done:
            action = self.choose_action(history) # 选择最大Q值出现频率最高的动作
            actions = [] # 没啥用，怀疑原本计划是在拼接箭头图片时使用，但是并没有使用，里面进行了重新计算
            for head_idx in range(self.config.n_ensemble):
                actions.append(self.choose_action(history, head_idx)) # 每个头选择一个动作

            # img = Image.fromarray(next_state)
            # frames.append(img)
            img = self.env.render(mode='rgb_array')
            img = Image.fromarray(img) # 将观察帧转换为图片
            if len(arrow_images) > 0:
                # 根据动作选择结果，加载对应的箭头图片
                img2 = arrow_images[action]
                # 水平拼接图片
                img = get_concat_h(img, img2)

                if self.config.n_ensemble > 1:
                    # 如果有多个头，则将每个头选择的动作对应的箭头图片垂直拼接
                    merge_img = None
                    for head_idx in range(self.config.n_ensemble):
                        # 先得到每个头选择的动作对应的箭头图片进行垂直拼接
                        action_img = arrow_images[self.choose_action(history, head_idx)]
                        merge_img = get_concat_v(merge_img, action_img)
                    # 然后将垂直拼接的图片和观察图片进行水平拼接，显示所有头的选择结果
                    img = get_concat_h(img, merge_img)

            frames.append(img)


            if terminal: ## There is error when it is just started. So do action = 1 at first
               action = 1
            next_state, reward, done, life = self.env.step(action)
            history.push(next_state) # 更新观察样本

            score += reward
            life = life['ale.lives']
            count = count + 1

            ## Terminal options
            if life < last_life:
                terminal = True
            else:
                terminal = False
            last_life = life
        self.env.close()
        # 将评估的结果保存为GIF图片
        frames[0].save(os.path.join(self.out_dir, 'Breakout_result.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=0.0001)
        print("save picture -- Breakout_result.gif")
        print("score", score)
        print("count", count)


    def train(self):
        # 训练模型
        progress_bar = tqdm(range(self.start_steps, self.max_steps)) # 构建整体的训练进度
        state = self.env.reset() # 重置环境
        # 构建历史观察样本处理对象，包含裁剪、缩放、图片多通道合并二值化、帧堆叠，也就是观察预处理增强
        history = historyDataset(self.history_size, state, self.crop_flag)
        done = False

        ##Report
        train_scores = deque(maxlen=10)
        train_lengths = deque(maxlen=10)
        episode = 0
        max_score = 0 # 评估的最大分数

        ##If it is done everytime init value
        train_score = 0 # 当前游戏回合的得分
        train_length = 0 # 当前游戏回合的长度
        last_life = 0 # 记录上一次的生命数
        terminal = True

        ## number of ensemble
        heads = list(range(self.n_ensemble))
        active_head = heads[0] # todo 作用

        try:
            for step in progress_bar:
                # 训练开始

                ## model update
                if step > self.learn_start and step % self.target_update == 0:
                    # 每隔一段时间将动作策略模型的参数复制到目标动作策略模型
                    self.target_model.load_state_dict(self.policy_model.state_dict())

                ## game is over
                if done:
                    # todo
                    np.random.shuffle(heads)
                    active_head = heads[0]

                    state = self.env.reset()
                    history = historyDataset(self.history_size, state, self.crop_flag)
                    train_scores.append(train_score)
                    train_lengths.append(train_length)
                    episode += 1

                    ##If it is done everytime init value
                    train_score = 0
                    train_length = 0
                    last_life = 0
                    terminal = True

                # 选择动作，这里的动作预测头使用的是active_head
                # todo active_head是怎么选择的
                action = self.choose_action(history, active_head, self.get_epsilon(step))
                if terminal: ## There is error when it is just started. So do action = 1 at first
                    # 因为Breakout游戏的特殊性，游戏开始时需要执行一个动作1才能真正开始
                    # 而游戏刚开始或者者生命值减少时，terminal会被置为True
                    # 这个时候就必须执行动作1才能继续游戏
                    action = 1
                next_state, reward, done, life = self.env.step(action)
                state = history.get_state() # 获取当前的环境观察样本
                history.push(next_state)
                next_state = history.get_state() # 获取执行动作后的下一个环境观察样本，这里是将实际执行的帧堆叠等处理后的样本作为实际的观察，因为他没有使用gym的环境包装
                life = life['ale.lives'] # 获取还剩余的生命数
                train_length = train_length + 1 

                ## Terminal options 判断生命是否丢失，如果丢失则判定为中断
                # 因为对于breakout游戏，中断了需要执行动作1才能继续游戏
                if life < last_life:
                    terminal = True
                else :
                    terminal = False
                last_life = life

                # 将当前的观察样本、动作、奖励、下一个观察样本、是否中断等存储到经验回放池
                self.memory.push(state, action, next_state, reward, done, life, terminal)
                if step > self.learn_start and step % self.update_freq == 0:
                    # 达到了起始训练的步数，并且到了训练的更新频率，则进行模型训练
                    self.replay(self.batch_size)

                train_score = train_score + reward # 累计当前回合的得分

                if step > self.eval_steps and step % self.eval_freq == 0:
                    # 进入模型评估阶段
                    train_mean_score = np.mean(train_scores) # 计算最近10个回合的平均得分
                    train_mean_length = np.mean(train_lengths) # 计算最近10个回合的平均长度
                    self.train_score_memory.append(train_mean_score)
                    self.train_length_memory.append(train_mean_length)

                    # train_score: 将最近10个回合的平均得分存储起来，覆盖之前的
                    # train_length: 将最近10个回合的平均长度存储起来，覆盖之前的
                    save_numpy(self.train_score_memory, self.out_dir, 'train_score')
                    save_numpy(self.train_length_memory, self.out_dir, 'train_length_memory')

                    # 评估模型，得到评估得分和评估执行的步数
                    valid_score, valid_length = self.valid_run()
                    self.test_score_memory.append(valid_score)
                    self.test_length_memory.append(valid_length)

                    # 存储评估得分和评估步数
                    save_numpy(self.test_score_memory, self.out_dir, 'test_score')
                    save_numpy(self.test_length_memory, self.out_dir, 'test_length_memory')

                    if valid_score >= max_score:
                        # 如果评估的分数大于之前的最大分数，则保存当前模型
                        # 不过这里只是保存模型，没有保存优化器等其他状态，所以不是可持续化训练的保存方式
                        max_score = valid_score
                        save_model(self.policy_model, self.out_dir)
                    
                    # 更新训练进度信息
                    progress_bar.set_postfix_str(
                        '[Episode %s] - train_score : %.2f, test_score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode,
                                                                                                                    train_mean_score,
                                                                                                                    valid_score,
                                                                                                                    max_score,
                                                                                                                    self.get_epsilon(step)))
                    
                    # # 打印训练的进度 日志
                    logging.debug(
                        '[Episode %s] - train_score : %.2f, test_score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode,
                                                                                                                    train_mean_score,
                                                                                                                    valid_score,
                                                                                                                    max_score,
                                                                                                                    self.get_epsilon(step)))
        except Exception as e:
            # Get current system exception
            # 如果出现异常，打印异常信息、堆栈
            ex_type, ex_value, ex_traceback = sys.exc_info()

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            logging.warning("Exception type : %s " % ex_type.__name__)
            logging.warning("Exception message : %s" % ex_value)
            for trace in trace_back:
                logging.warning("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                trace[0], trace[1], trace[2], trace[3]))


if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and config.device in ["gpu",'cuda'] else "cpu")
    config.device = device
    agent = DQNSolver(config)
    if config.mode == "train":
        agent.train()
    if config.mode =="test":
        # 测试模型，需要确保已经有预训练模型
        if config.pretrained_dir is None:
            raise ValueError(
                "평가를 하려면 pretrained_dir 에 저장된 모델을 넣어야 합니다. {}".format(
                    config.pretrained_dir
                )
            )
        agent.render_policy_net()