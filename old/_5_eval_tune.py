from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np
from constants import HAZARD_P, SLIP_P, GROUP_SIZE, BS, DIR
from helpers import batchl, flat
from model_reward import model_reward
import gc

    
def eval_tuned(map_size, student, teacher, train_size, seed):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
        
    rng = np.random.default_rng(seed)
    test_size = 1000
    test_groups = rng.choice(100, size=test_size//100, replace=False)
    x = []
    for test_group in test_groups:
        data = np.load(f'{DIR}/data{map_size}/test/myopic_{teacher}_{test_group}.npz')
        x.append(data['x'])    
    x = np.concatenate(x)
    start = rng.integers(map_size**2)
    x = x[start::map_size**2]
    if len(x) < test_size:
        raise Exception
    x = x[:test_size]
    
    model = tf.keras.models.load_model(f'{DIR}/models{map_size}/{0}_{teacher}_{train_size}_{seed%5}.keras')
    # model = tf.keras.models.load_model(f'{DIR}/models{map_size}/myopic_{teacher}.keras')
    all_results = []
    for si, state in enumerate(x):
        if si%100 == 0:
            print(si, flush=True)
        all_results.append(model_reward(map_size, model, state))
        
    with open(f'output/res_curr.txt', 'a') as f:
        print(f'{map_size},{student},{teacher},{train_size},{test_size},{seed},{np.mean(all_results)}', file=f)




if __name__=='__main__':

#     import sys
#     map_size = int(sys.argv[1])
#     student = int(sys.argv[2])
#     teacher = int(sys.argv[3])
#     train_size = int(sys.argv[4])
    
#     if len(sys.argv) >= 6:
#         offset = int(sys.argv[5])
#     else:
#         offset = 0
        
#     iterables = []
#     for seed in range(offset, offset+5):
#         iterables += [(map_size,student,teacher,train_size,seed)]
#     process_map(eval_tuned, *zip(*iterables), chunksize=1, max_workers=5)
        
        
    
    map_size = 8
    student = 0
    offset = 0
        
    teachers = range(1,2)
    train_sizes = 1000 * np.array([2,4,8,16,32,64,128,256,512,1024])
    seeds = range(offset, offset+5)
    
    iterables = []
    for seed in seeds:
        for teacher in teachers:
            for train_size in train_sizes:
                iterables += [(map_size,student,teacher,train_size,seed)]
                
    process_map(eval_tuned, *zip(*iterables), chunksize=1, max_workers=20)