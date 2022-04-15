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
        
        # Q table update
        
        # update state 
        
        # find wrong dropouts
        
        if done:
            break
        