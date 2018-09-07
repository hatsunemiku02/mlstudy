import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import namedtuple
from CNN_DQN_TORCH import CNN,CNN_DQNWarp,Transition
import matplotlib.pyplot as plt

ACTION_SPACE = 11
MEMORY_SIZE = 200



resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

if torch.cuda.is_available():
    print("use gpu, good!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_screen(_env):

    screen = _env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    # Strip off the top and bottom of the screen
    screen = screen[:, 100:400,100:400]
   
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

def train(_RL,_env):
    total_steps = 0
    _env.reset()

    last_screen = get_screen(_env)
    current_screen = get_screen(_env)
    state = current_screen - last_screen

    #plt.imshow(last_screen.detach().cpu().numpy().transpose((1,2,0)))
    #plt.show()

    while True:        
        action = RL.choose_action(state)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = _env.step(np.array([f_action]))

        last_screen = current_screen
        current_screen = get_screen(_env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        
        #if reward<-10:
          #  print(reward)
        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        _RL.store_transition(state, action,next_state, reward)

        if total_steps > MEMORY_SIZE :   # learning
            _RL.learn()

        #if total_steps - MEMORY_SIZE > 20000:   # stop game
        if done:
            #_RL.plot_q()
            #_RL.plot_cost()
            break
        state = next_state
        total_steps += 1
    return _RL.q

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)

    nn_evl = CNN(
        ACTION_SPACE,128 ,
    ).to(device)

    nn_tar = CNN(
        ACTION_SPACE,128 ,
    ).to(device)

    RL = CNN_DQNWarp(nn_evl,nn_tar,ACTION_SPACE, 3,
                      learning_rate=0.005,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=MEMORY_SIZE,
                      e_greedy_increment=0.0001,
                      device = device
                      )
    train(RL,env)