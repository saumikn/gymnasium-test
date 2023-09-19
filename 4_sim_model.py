from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np

from constants import MAP_SIZE, HAZARD_P, SLIP_P, MYOPIC, SEEDS


def model_reward(myo, state):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')
    
    env = MDPEnv(map_size=MAP_SIZE, hazard_p=0.3, slip_p=0.4)
    env.reset(state)
    
    agents = []
    grids = []
    target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if (i,j) == target or (i,j) in hazards:
                continue
            state = ((i,j), target, hazards, rand)
            env.reset(state)
            agents.append((i,j))
            grids.append(env.get_grid())
            
    model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models/myopic_{myo}.keras')
    actions = model.predict(np.array(grids)).argmax(axis=1)
    trans = dict(zip(agents, actions))
    
    rewards = []
    for i in range(100):
        env.reset(state)
        reward_sum = 0
        for step in range(100):
            action = trans[env.state['agent']]
            _, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break
        rewards.append(reward_sum)
    return np.mean(rewards)



def get_states():
    filename = f'/storage1/fs1/chien-ju.ho/Active/gym/data/myopic_{1}.gzip'
    with gzip.GzipFile(filename, 'rb') as f:
        res = pickle.loads(f.read())
        grids, actions = zip(*[(g,a) for _,g,a,_ in res])
        grids, actions = np.array(grids), np.array(actions)
        split = int(len(grids)*0.8)
        x_train, x_test = grids[:split], grids[split:]
        y_train, y_test = actions[:split], actions[split:]
        x_states = x_test[np.random.choice(len(x_test), size=1000, replace=False)]
    return x_states

x_states = get_states()

print('got states')


for myo in MYOPIC:
    myos = [myo for _ in x_states]
    res = process_map(model_reward, myos, x_states)
    print(myo, np.mean(res), np.std(res))