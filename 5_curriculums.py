from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np
from constants import *
from helpers import batchl, flat

def curr(MAP_SIZE, student, teacher, train, seed):
    test = 1000
    bs = 32
    
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{MAP_SIZE}/myopic_{student}.keras')
    # K.set_value(model.optimizer.learning_rate, 0.0001)
    
    rng = np.random.default_rng(seed)
    train_group, test_group = rng.choice(93, size=2, replace=False)
           
    if train > 0:
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{MAP_SIZE}/test/myopic_{teacher}_{train_group}.npz')
        x, y = data['x'], data['y']
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]

        model.fit(x[:train*bs], y[:train*bs], batch_size=bs, verbose=0)
    
    all_results = []
    
    data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{MAP_SIZE}/test/myopic_{teacher}_{train_group}.npz')
    x, y = data['x'], data['y']
    start = rng.integers(15)
    x, y = x[start::15], y[start::15]
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]

    for si, state in enumerate(x[:test]):
        # print(f'{si}/{test}', end='\r')
        env = MDPEnv(map_size=MAP_SIZE, hazard_p=0.3, slip_p=0.4)
        env.reset(state)

        agents = []
        grids = []
        states = []
        target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
        hazard_grid = env.get_grid()[2]
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if (i,j) == target or (i,j) in hazards:
                    continue
                if not is_valid(hazard_grid, (i,j), target):
                    continue
                state = ((i,j), target, hazards, rand)
                env.reset(state)
                agents.append((i,j))
                grids.append(env.get_grid())
                states.append(state)

        actions = model.predict(np.array(grids)).argmax(axis=1)
        trans = dict(zip(agents, actions))

        visited = {}

        values = np.zeros((MAP_SIZE,MAP_SIZE))

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

        all_rewards = []
        for (i,j) in agents:
            all_rewards.append(values[i,j])
            
        all_results.append(np.mean(all_rewards))
        
    with open(f'output/res{MAP_SIZE}.txt', 'a') as f:
        print(f'{student},{teacher},{train},{test},{seed},{np.mean(all_results)}', file=f)

    return np.mean(all_results)

if __name__=='__main__'
iterables = []
MAP_SIZE = 8
for seed in range(0,100): # 0,200
    for student in [2]:
        for teacher in range(3,21):
            # for train in [1, 3, 10, 30, 100, 300, 1000]:
            for train in [20, 40, 60, 80]:
                iterables.append((MAP_SIZE,student,teacher,train,seed))
res = process_map(curr, *zip(*iterables), chunksize=1)