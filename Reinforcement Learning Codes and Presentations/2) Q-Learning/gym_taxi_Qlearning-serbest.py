from os import stat
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env

# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])  # env.observation_space.n bize state sayısını verir. env.action_space.n bize env'deki action sayısını verir.

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting metrix
reward_list = []
dropout_list = []

episode_number = 10000
for i in range(1,episode_number):
    
    # initialize enviroment
    state = env.reset()
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # exploit vs explore to find action
        # %10 explore, %90 exploit
        # 0 ile 1 arasında rastgele üretilen bir sayı epsilondan küçük ise rastgele bir action seçilir yani explore(keşif) yapılır
        # eğer değilse o zaman q table içerisinde o state için en büyük değerli action seçilir.
        if random.uniform(0,1) < epsilon: # 0 ile 1 arasında rastgele üretilen bir sayı epsilondan küçük ise rastgelebir
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        # action process and take reward / observation
        # env.step() fonksiyonu action 'ı uygulayan fonksiyondur. geriye 4 değer döndürmektedir. 
        # next_state action uygulandığında gidilecek olan state, reward ödül, done ise dropout yapılıp yapılmadığı.
        next_state, reward, done, _ = env.step(action)
        
        # Q Learning Function
        old_value = q_table[state, action]
        next_max = np.max( q_table[next_state] )
        
        next_value = ( 1-alpha )*old_value + alpha * (reward + gamma * next_max)
        
        # Q table update
        q_table[ state, action ] = next_value
        # update state 
        state = next_state
        # find wrong dropouts
        if reward == -10:
            dropouts +=1
            
        if done:
            break
        reward_count += reward

    if i%10 == 0:
        dropout_list.append( dropouts )
        reward_list.append( reward_count )
        print( "Episode : {}, reward {}, wrong dropout {}".format(i, reward_count,dropouts) )

# %% visualize
fig,axs = plt.subplots( 1, 2 )
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()

