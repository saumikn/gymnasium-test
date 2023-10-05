from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np
from constants import HAZARD_P, SLIP_P, GROUP_SIZE, BS
from helpers import batchl, flat
from model_reward import model_reward


def curr(map_size, student, teacher, train_size, seed):    
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{student}.keras')
    # K.set_value(model.optimizer.learning_rate, 0.0001)
    
    rng = np.random.default_rng(seed)
    train_groups = rng.choice(700, size=15, replace=False)
    x, y = [], []
    for train_group in train_groups:
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/train/myopic_{teacher}_{train_group}.npz')
        x.append(data['x'])
        y.append(data['y'])
    x, y = np.concatenate(x), np.concatenate(y)
    print(x.shape, y.shape, seed)
    if train_size:
        model.fit(x[:train_size*BS], y[:train_size*BS], batch_size=BS, verbose=0)
        
    test_size = 1000
    test_groups = rng.choice(700, size=12, replace=False)
    x = []
    for test_group in test_groups:
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{test_group}.npz')
        x.append(data['x'])
    
    x = np.concatenate(x)
    start = rng.integers(map_size**2)
    x = x[start::map_size**2]
    x = x[:test_size]
    
    print(x.shape, seed)
    

    all_results = []
    for si, state in enumerate(x):
        print(si, end='\r')
        all_results.append(model_reward(model, map_size, state))
        
        
    with open(f'output/res_curr.txt', 'a') as f:
        print(f'{map_size},{student},{teacher},{train_size},{test_size},{seed},{np.mean(all_results)}', file=f)

#     return np.mean(all_results)

if __name__=='__main__':

    import sys
    map_size = int(sys.argv[1])
    student = int(sys.argv[2])
    teacher = int(sys.argv[3])
    train_size = int(sys.argv[4])
    
    if len(sys.argv) >= 6:
        offset = int(sys.argv[5])
    else:
        offset = 0
        
    iterables = [(map_size,student,teacher,train_size,seed) for seed in range(offset, offset+100)]
    res = process_map(curr, *zip(*iterables), chunksize=1, max_workers=5)