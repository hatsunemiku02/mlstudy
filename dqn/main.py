import gym
import numpy as np
from DQN_torch import Net,DQNWarp

ACTION_SPACE = 11
MEMORY_SIZE = 3000

def train():
    total_steps = 0
    observation = env.reset()
    while True:
        if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        #if reward<-10:
          #  print(reward)
        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE :   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            RL.plot_q()
            RL.plot_cost()
            break

        observation = observation_
        total_steps += 1
    return RL.q

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)

    nn_evl = Net(
        ACTION_SPACE,3,20 ,
    )

    nn_tar = Net(
        ACTION_SPACE,3,20 ,
    )

    RL = DQNWarp(nn_evl,nn_tar,ACTION_SPACE, 3,
                      learning_rate=0.005,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=MEMORY_SIZE,
                      e_greedy_increment=0.0010
                      )
    train()