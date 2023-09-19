import numpy as np
import pandas as pd
from mdp import MDPEnv, is_valid
from functools import cache, lru_cache
from collections import defaultdict

MAP_SIZE = 4  # Number of tiles of one side of the squared environment
HAZARD_P = 0.3  # Probability that square is hazadorous
SLIP_P = 0.4  # Probability of slipping on the ice
MYOPIC = list(range(1,8))

ITERATIONS = 10

import tensorflow as tf


def exp(seed):
    @cache
    def myopic(state, k=1):
        if k == 0:
            return None, 0

        env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P)
        
        results = []
        actions = list(range(env.action_space.n))
        np.random.shuffle(actions)
        for action in actions:
            env.reset(state)
            probs, next_agents = env.next_agents(action)
            
            reward_sum = 0
            for p, na in zip(probs, next_agents):
                _, reward, terminated, truncated, _ = env.step(next_agent = na)
                if terminated or truncated:
                    reward_sum += p * reward
                else:
                    _, next_reward = myopic(env.get_tuple(), k - 1)
                    reward_sum += p * (reward + next_reward)
            results.append((reward_sum, action))
            
            
        best_reward, best_action = max(results)
        return best_action, best_reward
        
        
        
#         results = defaultdict(list)
        
        
#         for _ in range(ITERATIONS):
#             env.reset(state)
#             action = env.action_space.sample()
#             _, reward, terminated, truncated, _ = env.step(action=action)

#             if terminated or truncated:
#                 results[action].append(reward)
#             else:
#                 _, next_reward = myopic(env.get_tuple(), k - 1)
#                 results[action].append(reward + next_reward)
#         env.reset(state)

#         results = {k: np.mean(v) for k, v in results.items()}
#         best_action = max(results, key=results.get)
#         best_reward = results[best_action]
#         return best_action, best_reward
    
    
    env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P, seed=seed)
    state, _ = env.reset()
    target = state['target']
    hazards = state['hazards']
    rand = state['rand']
    
    results = []
    
    hazard_grid = env.get_grid()[2]
    
    for m in MYOPIC:
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if (i,j) == target or (i,j) in hazards:
                    continue
                if not is_valid(hazard_grid, (i,j), target):
                    continue
                state = ((i,j), target, hazards, rand)
                env.reset(state)
                
                best_action, best_reward = myopic(state, k=m)
                results.append((m, env.get_grid(), best_action, best_reward))
                
    return results

def make_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(4, MAP_SIZE, MAP_SIZE)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model


def model_reward(model, state):
    env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P)
    
    
    reward_sum = []
    for i in range(10):
        env.reset(state)
        
        for step in range(100):
            grid = env.get_grid()[np.newaxis,:]
            action = model.predict(grid)
    