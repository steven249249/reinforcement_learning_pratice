"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate #*loss = target value - predicted value
EPSILON = 0.9               # greedy policy，也就是exploration，用來讓模型去選擇做並非q function 決定的東西，而是去做別的事讓q function可以取樣
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency ,target network是生成target value右式的東西,參數更新時只更新左邊的network
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
#unwrapped提供修改環境的辦法,也就是解除限制
env = env.unwrapped
N_ACTIONS = env.action_space.n #n個動作可選
N_STATES = env.observation_space.shape[0] #(4) state表示四種狀態
#車的位置（Cart Position）：表示車子在水平軸上的位置。
# 車的速度（Cart Velocity）：表示車子在水平軸上的速度。
# 棒子的角度（Pole Angle）：表示棒子相對垂直方向的角度。
# 棒子的角速度（Pole Angular Velocity）
# env.action_space.sample() 是 OpenAI Gym 環境中的一個方法，用於從動作空間中隨機選擇一個動作並返回。

# 根據不同的環境，動作空間可以是離散的或連續的。對於離散動作空間，env.action_space.sample() 將返回一個整數，該整數表示從動作空間中選擇的具體動作。例如，在 CartPole 環境中，動作空間是離散的，可能的動作是向左推（0）和向右推（1）。

# 對於連續動作空間，env.action_space.sample() 將返回一個連續動作，該動作是從動作空間中按某種分布（例如均勻分布）隨機取樣得到的。連續動作可以是一個數字或者一個向量
#具體取決於動作空間的形狀。


#如果動作空間是離散的(判斷是否為int,如果是連續的則有可能是向量而不是int)，則 ENV_A_SHAPE 為 0；如果動作空間是連續的，則 ENV_A_SHAPE 為取樣動作的形狀。請注意，這僅適用於連續動作空間具有形狀屬性的情況。
#例如，如果連續動作空間是二維的，則 env.action_space.sample().shape 可能返回 (2,)，表示該連續動作有兩個維度。比如往左往上移動各移動X之類的就會變成(x,y)
#此項用來在以下做陣列處理
#其實連續的也不太會用DQN處理
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# env = gym.make('CartPole-v0')   # 定义使用 gym 库中的那一个环境
# env = env.unwrapped # 不做这个会有很多限制

# print(env.action_space) # 查看这个环境中可用的 action 有多少个
# print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
# print(env.observation_space.high)   # 查看 observation 最高取值
# print(env.observation_space.low)    # 查看 observation 最低取值

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory，n_state 是 s s_1 ,2是a,r
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        #生成一個0~1之間的數 np.random.uniform(low=0.0, high=1.0, size=None)
        #greedy policy，也就是exploration，用來讓模型去選擇做並非q function 決定的東西，而是去做別的事讓q function可以取樣
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            #找出Q FUNCTION(Q model)所有行動中最大的q值
            #action會給values,indices,所以這裡是傳回indice
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        #replay buffer 的意思等於memory 
        #共2000筆(s,a,r,s_)可存
        
        #把這幾個東西拚成一列
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY #如果超過memory capicity,新的會把前面的蓋掉
        #使用冒號可以更清楚表達幾維，用self.memory[index]一樣
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update,target network的更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            #target讀取eval_network的參數
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        
        #第一個參數是從0到memory_capicity的範圍,第二個參數是取幾個,所以這裡的意思就是從memory取一個batch size的index出來
        #所以b_memory會變成二維的 batch_size * (N_STATE +1 + 1 +N_STATE)
        #b_s的大小就是 batch_size * N_state,其他同理
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)) 
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        #在維度=1的狀況下，從一個array拿出index的東西,給定一個index array，返回該index array的對應值
        
        # 假设有一个形状为 (3, 4) 的输入张量 input_tensor
        # input_tensor = torch.tensor([[1, 2, 3, 4],
        #                              [5, 6, 7, 8],
        #                              [9, 10, 11, 12]])
        # # 假设有一个形状为 (2, 3) 的索引张量 index
        # index = torch.tensor([[0, 2, 1],
        #                       [2, 1, 0]])  
        # # 在 dim=1 的维度上从 input_tensor 中按照 index 进行收集
        # output_tensor = input_tensor.gather(1, index)  
        # print(output_tensor)
        # tensor([[ 1,  3,  2],
        #         [ 7,  6,  5]])
        
        #給定memory中每個s所採取的行動作為index,net ouput出來的是 batch_size*action ,所以會拿出所有batch選擇的q(a),變成batch*1
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 每個batch的行動的q值
        
        #具体而言，detach() 方法用于创建一个新的张量，该张量没有与原始张量相关的梯度信息。这意味着对分离后的张量进行操作不会对原始张量的梯度产生影响,目的是不讓target_net更新
        #這邊不用gather是因為沒有對於s_的選擇動作,所以要自動用max去找(本來q learning就是這樣做的)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        #q_next.max是給max_values max_indices，這裡取value,view是reshape的意思
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    # 获取回合 i_episode 第一个 observation
    s = env.reset()
    ep_r = 0
    while True:
        print(N_STATES)
        env.render() # 刷新环境
        a = dqn.choose_action(s)

        # take action
        # next_state：下一個狀態（state），表示在執行動作 a 後環境的狀態。
        # reward：執行動作 a 後獲得的即時獎勵值。
        # done：一個布林值，表示是否達到終止狀態（例如，遊戲結束或目標達成）。
        # info：一個字典，提供額外的環境資訊（通常用於調試或監控）
        s_, r, done, info = env.step(a)
        
        # modify the reward
        # x：倒立擺的水平位置。
        # x_dot：倒立擺的水平速度。
        # theta：倒立擺的角度（相對於垂直方向）。
        # theta_dot：倒立擺的角速度。
        x, x_dot, theta, theta_dot = s_
        
        #x_threshold:倒立擺的水平位置 x 的臨界值。超過就輸
        #env.theta_threshold_radians 倒立擺的角度 theta 的臨界值
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_