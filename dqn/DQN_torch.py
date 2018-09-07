import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

np.random.seed(1)

class Net(nn.Module):

    def __init__(
        self,
        n_actions,
        n_features,
        n_l1count,
        ):
        super(Net, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_l1count = n_l1count

        # net work
        self.evl_fc1 = nn.Linear(self.n_features, n_l1count)
        init.constant_(self.evl_fc1.bias,0.1)
        init.normal_(self.evl_fc1.weight,0.0,0.3)
        self.evl_fc2 = nn.Linear(n_l1count, self.n_actions)
        init.constant_(self.evl_fc2.bias,0.1)
        init.normal_(self.evl_fc2.weight,0.0,0.3)

    def forward(self, x): 
        x = F.relu(self.evl_fc1(x))
        x = self.evl_fc2(x)
        return x
    

class DQNWarp:
    def __init__(
        self,
        nn_evl,
        nn_target,
        n_actions,
        n_features,
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


        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.optimizer = optim.RMSprop(self.nn_evl.parameters(), self.lr)
        self.cost_his = []

    def run_evl(self,x):   
        tensor = torch.from_numpy(x)
        output = self.nn_evl(tensor.float())
        test = output.detach().numpy()
        return output
    
    def run_tar(self,x):
        tensor = torch.from_numpy(x)
        return self.nn_target(tensor.float())

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            with torch.no_grad():
                actions_value = self.run_evl(observation)
            action = torch.Tensor.argmax(actions_value).detach().numpy()
            if not hasattr(self, 'q'):  # record action value it gets
                self.q = []
                self.running_q = 0
           # print("choose_action:"+str(actions_value.detach().numpy()))
            self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value.detach().numpy())
            self.q.append(self.running_q)

        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
           self.nn_target.load_state_dict(self.nn_evl.state_dict())
           print("replace target values\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            #print("learn 1 :")
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            #print("learn 2 :")
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

       
        q_next = self.run_tar(batch_memory[:, -self.n_features:])
        q_eval = self.run_evl(batch_memory[:, :self.n_features])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()
     
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        batch_index_t = torch.from_numpy(batch_index)
        eval_act_index_t = torch.from_numpy(eval_act_index)
        reward_t = torch.from_numpy(reward)
       
       # q_target[batch_index, eval_act_index] = reward + self.gamma * torch.Tensor.max(q_next, dim=1)
        max_qnext,max_index = torch.Tensor.max(q_next, dim=1)
        mpin = max_index.numpy() 
        mpvalue = max_qnext.detach().numpy()
        q_target[batch_index, eval_act_index] = reward_t.float() + self.gamma * max_qnext
       
        self.optimizer.zero_grad() 
        #criterion =
        
        loss = F.smooth_l1_loss(q_eval,q_target.detach())
        loss.backward()
        for param in self.nn_evl.parameters():
	        param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self.cost_his.append(loss.detach().numpy())

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


