from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np

# from constants import MAP_SIZE, HAZARD_P, SLIP_P, MYOPIC, SEEDS
from constants import HAZARD_P, SLIP_P, GROUP_SIZE


def model_reward(map_size, teacher, state):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')
    
    # env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
    # env.reset(state)
    
#     agents = []
#     grids = []
#     states = []
#     target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
#     hazard_grid = env.get_grid()[2]
#     for i in range(map_size):
#         for j in range(map_size):
#             if (i,j) == target or (i,j) in hazards:
#                 continue
#             if not is_valid(hazard_grid, (i,j), target):
#                 continue
#             state = ((i,j), target, hazards, rand)
#             env.reset(state)
#             agents.append((i,j))
#             grids.append(env.get_grid())
#             states.append(state)
            
#     model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{teacher}.keras')
#     actions = model.predict(np.array(grids)).argmax(axis=1)
#     trans = dict(zip(agents, actions))
    
#     rewards = []
#     for i in range(100):
#         env.reset(state)
#         reward_sum = 0
#         for step in range(100):
#             action = trans[env.state['agent']]
#             _, reward, terminated, truncated, _ = env.step(action)
#             reward_sum += reward
#             if terminated or truncated:
#                 break
#         rewards.append(reward_sum)
    
#     return np.mean(rewards)


    # all_rewards = []
    # for si, state in enumerate(x[:test]):
    # print(f'{si}/{test}', end='\r')
    
    
    env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
    env.reset(state)

    agents = []
    grids = []
    states = []
    target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
    hazard_grid = env.get_grid()[2]
    for i in range(map_size):
        for j in range(map_size):
            if (i,j) == target or (i,j) in hazards:
                continue
            if not is_valid(hazard_grid, (i,j), target):
                continue
            state = ((i,j), target, hazards, rand)
            env.reset(state)
            agents.append((i,j))
            grids.append(env.get_grid())
            states.append(state)
    model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{teacher}.keras')
    actions = model.predict(np.array(grids)).argmax(axis=1)
    trans = dict(zip(agents, actions))
    visited = {}
    values = np.zeros((map_size,map_size))

    @cache
    def next_agents(agent):
        env.reset(state)
        env.state['agent'] = agent
        action = trans[agent]
        probs, next_agents = env.next_agents(action)
        res = []
        for prob, next_agent in zip(probs, next_agents):
            env.reset(state)
            env.state['agent'] = agent
            _, reward, terminated, truncated, _ = env.step(next_agent=next_agent)
            res.append((prob, next_agent, reward))
        return res

    for vi in range(100):
        next_values = np.zeros_like(values)
        for i, j in agents:
            for prob, (i2,j2), reward in next_agents((i,j)):
                # print(prob, (i,j), (i2,j2), reward)
                next_values[i][j] += prob * (values[i2,j2] + reward)
        values = next_values

    return np.mean([values[i,j] for (i,j) in agents])


    
    # def step(agent):
    #     action = trans[env.state['agent']]
    #     _, reward, terminated, truncated, _ = env.step(action)
    #     reward_sum += reward
    #     if terminated or truncated:
    #         break
            
#     @cache
#     def get_reward(state):
#         env.reset(state)
#         action = trans[env.state['agent']]
#         probs, next_agents = env.next_agents(action)
#         reward = 0
#         for prob, next_agent in zip(probs, next_agents):
#             next_state, reward, terminated, truncated, _ = env.step(next_agent=next_agent)
#             if terminated or truncated:
#                 reward += (prob * reward)
#             else:
#                 next_reward = get_reward(env.get_tuple())
#                 reward += (prob * (reward + next_reward))
#         return reward
                
#     all_rewards = []
#     for state in states:
#         all_rewards.append(get_reward(state))

#     return np.mean(all_rewards)


# def get_states():
#     filename = f'/storage1/fs1/chien-ju.ho/Active/gym/data/myopic_{1}.gzip'
#     with gzip.GzipFile(filename, 'rb') as f:
#         res = pickle.loads(f.read())
#         grids, actions = zip(*[(g,a) for _,g,a,_ in res])
#         grids, actions = np.array(grids), np.array(actions)
#         split = int(len(grids)*0.8)
#         x_train, x_test = grids[:split], grids[split:]
#         y_train, y_test = actions[:split], actions[split:]
#         x_states = x_test[np.random.choice(len(x_test), size=1000, replace=False)]
#     return x_states

# x_states = get_states()

# print('got states')

if __name__ == '__main__':
    import sys
    map_size = int(sys.argv[1])
    teacher = int(sys.argv[2])
    
    x_states = []
    for group in range(10):
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{group}.npz')
        x = data['x']
        x_states.append(x)
    x_states = np.concatenate(x_states)
    
    x_states = x_states[::map_size**2]
    x_states = x_states[:1000]
    
    # print(x_states.shape)
    
    map_sizes = [map_size for _ in x_states]
    teachers = [teacher for _ in x_states]
    
    res = process_map(model_reward, map_sizes, teachers, x_states, disable=True)
    
    with open('output/eval_sim.txt', 'a') as f:
        print(f'{map_size},{teacher},{np.mean(res)}', file=f, flush=True)
