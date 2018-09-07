import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as T
from collections import namedtuple

np.random.seed(1)


class CNN(nn.Module):
    def __init__(
        self,
        n_actions,
        n_l1count,
        ):
        super(CNN, self).__init__()
        self.n_actions = n_actions
        self.n_l1count = n_l1count

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(n_l1count, n_actions)

    def forward(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class CNN_DQNWarp:
    def __init__(
        self,
        nn_evl,
        nn_target,
        n_actions,
        n_features,
        device,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.99,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None
        ):
        self.nn_evl = nn_evl
        self.nn_target = nn_target
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.epsilon_increment = e_greedy_increment
        self.device = device

        # initialize zero memory [s, a, r, s_]
        self.memory =  []
        self.optimizer = optim.RMSprop(self.nn_evl.parameters(), self.lr)
        self.cost_his = []

    def run_evl(self,x):   
        output = self.nn_evl(x)
        return output
    
    def run_tar(self,x):
        return self.nn_target(x)

    def store_transition(self, *agrs):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition =  Transition(*agrs)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size

        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[index] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            with torch.no_grad():
                actions_value = self.run_evl(observation.to(self.device))
            cpu_action =  torch.Tensor.argmax(actions_value).detach().cpu()
            action = cpu_action.numpy()
            if not hasattr(self, 'q'):  # record action value it gets
                self.q = []
                self.running_q = 0
           # print("choose_action:"+str(actions_value.detach().numpy()))
           # self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value.detach().numpy())
           # self.q.append(self.running_q)

        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
           self.nn_target.load_state_dict(self.nn_evl.state_dict())
           print("replace target values\n")

        # sample batch memory from all memory
        #if self.memory_counter > self.memory_size:
            #print("learn 1 :")
          #  sample_index = np.random.choice(self.memory_size, size=self.batch_size)
      #  else:
            #print("learn 2 :")
         #   sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        selected_transitions = random.sample(self.memory, self.batch_size)
        
       
        batch = Transition(*zip(*selected_transitions))

        nextdata = batch.next_state
        nextdata_torch = torch.cat(batch.next_state)
        
        evaldata = batch.state
        evaldata_torch = torch.cat(evaldata)
       
        q_next = self.run_tar(nextdata_torch.to(self.device))
        q_eval = self.run_evl( evaldata_torch.to(self.device))

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()
     
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #eval_act_index = batch_memory[:, self.n_features].astype(int)
        eval_act_index = batch.action
        #reward = batch_memory[:, self.n_features + 1]
        reward = batch.reward

        batch_index_t = torch.from_numpy(batch_index)
        eval_act_index_np = np.array(eval_act_index)
        eval_act_index_t = torch.from_numpy(eval_act_index_np)
        #eval_act_index_t = torch.cat(eval_act_index)
        reward_t = torch.tensor(reward).to(self.device)
       
       # q_target[batch_index, eval_act_index] = reward + self.gamma * torch.Tensor.max(q_next, dim=1)
        max_qnext,max_index = torch.Tensor.max(q_next, dim=1)
       # mpin = max_index.numpy() 
       # mpvalue = max_qnext.detach().numpy()
        q_target[batch_index_t.long(), eval_act_index_t.long()] = reward_t.float()+ self.gamma * max_qnext.to(self.device)
       
        self.optimizer.zero_grad() 
        #criterion =
        
        loss = F.smooth_l1_loss(q_eval,q_target.detach())
        loss.backward()
        for param in self.nn_evl.parameters():
	        param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
      #  self.cost_his.append(loss.detach().numpy())

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def plot_q(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.q)), self.q)
        plt.ylabel('q-value')
        plt.xlabel('training steps')
        plt.show()


