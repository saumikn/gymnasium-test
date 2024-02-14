import numpy as np
np.set_printoptions(linewidth=1000, suppress=True)
import pandas as pd
import seaborn as sns
from constants import DIR, HAZARD_P, SLIP_P
from functools import cache
from mdp import MDPEnv, is_valid

from itertools import product
from scipy.spatial.distance import cdist

def model_reward(map_size, model, state, max_iters=100000, gamma=1, verbose=False):
        
    env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
    env.reset(state)

    agents = []
    grids = []
    states = []
    target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
    hazard_grid = env.get_grid()[2]
    rand_grid = env.get_grid()[3]
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
    
    
    if isinstance(model, int):
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.get_logger().setLevel('ERROR')  
        # model = tf.keras.models.load_model(f'{DIR}/models{map_size}/myopic_{model}.keras')
        # actions = model.predict(np.array(grids)).argmax(axis=1)
    elif isinstance(model, np.ndarray):
        actions = model
    else: # Is keras model already
        actions = model.predict(np.array(grids)).argmax(axis=1)
        
    trans = dict(zip(agents, actions))
    visited = {}

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
    
    values = np.zeros((map_size,map_size))
    for vi in range(max_iters):
        # print(vi, end='\r')
        next_values = np.zeros_like(values)
        for i, j in agents:
            for prob, (i2,j2), reward in next_agents((i,j)):
                # print(prob, (i,j), (i2,j2), reward)
                # next_values[i][j] += prob * (gamma*values[i2,j2] + reward)
                next_values[i][j] += prob * (gamma*values[i2,j2] + reward - rand_grid[i2,j2])
        if np.abs(values-next_values).sum() < 0.0001:
            break
        values = next_values
    # print()
        
    avg_value = np.mean([values[i,j] for (i,j) in agents])
        
    if verbose == False:
        return np.mean([values[i,j] for (i,j) in agents])
    
    dirs = np.zeros((map_size, map_size), dtype=str)
    dirs[:] = ' '
    key = {0:'↓', 1:'→', 2:'↑', 3:'←'}
    for i, j in agents:
        dirs[i,j] = key[trans[(i,j)]]
        
        
    val_reward = values
    val_reward += 10*env.get_grid()[1]
    val_reward += -1*env.get_grid()[2]
    val_reward += env.get_grid()[3]
    
    return avg_value, dirs, val_reward


def geti(x, start):
    i0, i1 = start, start
    for i1 in range(i0, -2, -1):
        if (x[i0][3]!=x[i1][3]).mean():
            break
    i0 = i1+1
    for i1 in range(i0, len(x)+1):
        if i1==len(x):
            break
        if (x[i0][3]!=x[i1][3]).mean():
            break

    return i0, i1

def reward_eval(map_size, student, teacher, ds, DIR=DIR, num_dense=4, num_nodes=1024, ts=1000, seed=0, markers=1):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    skip = map_size**2
    test_size = ts * skip
    
    x, y = [], []
    for test_group in range(test_size//3000+1):
        data = np.load(f'{DIR}/data{map_size}/test/myopic_{teacher}_{test_group}.npz')
        _x = data['x']
        markers_arr = np.zeros((len(_x), map_size*2-1, map_size, map_size))
        if markers:
            for j in range(len(_x)):
                xj = _x[j]
                mj = markers_arr[j]
                xa = np.argwhere(xj[0])
                xb = np.array(list(product(*[range(k) for k in xj[0].shape])))
                dists = cdist(xa, xb, metric='cityblock')
                idxs = np.hstack([dists.T, xb]).astype(int)
                mj[tuple(idxs.T)] = 1
            _x = np.concatenate((_x, markers_arr), axis=1)

        x.append(_x)   
        y.append(data['y'])
    x = np.concatenate(x)
    y = np.concatenate(y)
        
    avg_trues, avg_preds, avg_accs = [], [], []
    model = tf.keras.models.load_model(f'{DIR}/models_{num_dense}_{256}_{map_size}_{markers}/{student}_{teacher}_{ds}_{seed}.keras') 
    y_pred = model(x).numpy().argmax(axis=1)
    for start in range(skip,test_size,skip):
        i0, i1 = geti(x, start)
        # with tf.device("CPU"):
        #     test_dataset = tf.data.Dataset.from_tensor_slices(x[i0:i1])
        #     test_dataset = test_dataset.shuffle(100000, seed=seed).batch(i1-i0)
        avg_preds.append(model_reward(map_size, y_pred[i0:i1], x[i0], gamma=1))
        avg_trues.append(model_reward(map_size, y[i0:i1], x[i0], gamma=1))
        avg_accs.append(np.mean(y_pred[i0:i1]==y[i0:i1]))

    return avg_trues, avg_preds, avg_accs

if __name__ == '__main__':
    import sys
    node = int(sys.argv[1])
    student = int(sys.argv[2])
    teacher = int(sys.argv[3])
    
    from tqdm.contrib.concurrent import process_map
    from constants import DIR
    
    map_size = 6
    ts = 100
    dense = 4
    
    res = []
    iterables = []
    for ds in [2000, 4000, 8000, 16000, 32000, 64000]:
        for seed in range(1):
            iterables.append([map_size, student, teacher, ds, DIR, dense, node, ts, seed])
    print(f'{node} {student} {teacher}')
    res = process_map(reward_eval, *zip(*iterables), chunksize=1, max_workers=10, leave=False)
    res = np.array(res).mean(axis=2)
    
    with open('output/3_eval_models.txt', 'a') as f:
        for i, r in zip(iterables, res):
            map_size, student, teacher, ds, DIR, dense, node, ts, seed = i
            print(f'{student},{teacher},{map_size},{ds},{node},{r[0]},{r[1]},{r[2]}', file=f, flush=True)