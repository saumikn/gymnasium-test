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


save_points = [1,2,3,4,5,6,7,8,9,
               10,20,30,40,50,60,70,80,90,
               100,200,300,400,500,600,700,800,900,
               1000]

def tune_model(map_size, student, teacher, train_size, seed):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(f'{DIR}/models{map_size}/myopic_{student}.keras')

    rng = np.random.default_rng(seed)
    if train_size:    
        num_groups = int(np.ceil(train_size/1000))
        train_groups = rng.choice(1000, size=num_groups, replace=False)
        x, y = [], []
        for i, group in enumerate(train_groups):
            print(f'Training at {i}', end='\r')
            data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/train/myopic_{teacher}_{group}.npz')
            with tf.device("CPU"):
                train_dataset = tf.data.Dataset.from_tensor_slices((data['x'], data['y']))
                train_dataset = train_dataset.shuffle(10000, seed=seed).batch(BS).take(1000)
            model.fit(train_dataset, verbose=0)
            if i+1 in save_points:
                model.save(f'{DIR}/tmp/{map_size}_{student}_{teacher}_{(i+1)*1000}_{seed}.keras')
            # del train_dataset
            # gc.collect()
            
        # x, y = np.concatenate(x), np.concatenate(y)
        # if len(x) < train_size*BS:
        #     raise Exception
        # with tf.device("CPU"):
        #     train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        #     train_dataset = train_dataset.shuffle(10000).batch(BS).take(train_size)
        # model.fit(train_dataset)
        # model.fit(x[:train_size*BS], y[:train_size*BS], batch_size=BS, verbose=0)
        
#     model.save(f'{DIR}/tmp/{map_size}_{student}_{teacher}_{train_size}_{seed}.keras')
    
#     import gc
#     del x
#     del y
#     del train_dataset
#     gc.collect()

    
def eval_tuned(map_size, student, teacher, train_size, seed):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
        
    rng = np.random.default_rng(seed)
    test_size = 1000
    test_groups = rng.choice(700, size=10, replace=False)
    x = []
    for test_group in test_groups:
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{test_group}.npz')
        x.append(data['x'])    
    x = np.concatenate(x)
    start = rng.integers(map_size**2)
    x = x[start::map_size**2]
    if len(x) < test_size:
        raise Exception
    x = x[:test_size]
    
    model = tf.keras.models.load_model(f'{DIR}/tmp/{map_size}_{student}_{teacher}_{train_size}_{seed}.keras')
    all_results = []
    for si, state in enumerate(x):
        print(si, end='\r')
        all_results.append(model_reward(model, map_size, state))
        
        
    # with open(f'output/res_curr.txt', 'a') as f:
    #     print(f'{map_size},{student},{teacher},{train_size},{test_size},{seed},{np.mean(all_results)}', file=f)





def curr(map_size, student, teacher, train_size, seed):
    rng = np.random.default_rng(seed)
    model = tune_model(map_size, student, teacher, train_size, seed, rng)
    return model
#     import tensorflow as tf
#     from tensorflow.keras import backend as K
#     for gpu in tf.config.list_physical_devices('GPU'):
#         tf.config.experimental.set_memory_growth(gpu, True)
#     tf.get_logger().setLevel('ERROR')
#     model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{student}.keras')
#     # K.set_value(model.optimizer.learning_rate, 0.0001)
    
#     rng = np.random.default_rng(seed)
#     train_groups = rng.choice(700, size=5, replace=False)
#     x, y = [], []
#     for train_group in train_groups:
#         data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/train/myopic_{teacher}_{train_group}.npz')
#         x.append(data['x'])
#         y.append(data['y'])
#     x, y = np.concatenate(x), np.concatenate(y)
#     print(x.shape, y.shape, seed)
#     if train_size:
#         model.fit(x[:train_size*BS], y[:train_size*BS], batch_size=BS, verbose=0)
        
#     test_size = 1000
#     test_groups = rng.choice(700, size=12, replace=False)
#     x = []
#     for test_group in test_groups:
#         data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{test_group}.npz')
#         x.append(data['x'])
    
#     x = np.concatenate(x)
#     start = rng.integers(map_size**2)
#     x = x[start::map_size**2]
#     x = x[:test_size]
    
#     print(x.shape, seed)
    

#     all_results = []
#     for si, state in enumerate(x):
#         print(si, end='\r')
#         all_results.append(model_reward(model, map_size, state))
        
        
#     with open(f'output/res_curr.txt', 'a') as f:
#         print(f'{map_size},{student},{teacher},{train_size},{test_size},{seed},{np.mean(all_results)}', file=f)

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
        
    iterables = [(map_size,student,teacher,train_size,seed) for seed in range(offset, offset+10)]
    
    
    import time
    
    st = time.perf_counter()
    
    process_map(tune_model, *zip(*iterables), chunksize=1, max_workers=10)
    
    # print(time.perf_counter() - st)
    
    # process_map(eval_tuned, *zip(*iterables), chunksize=1, max_workers=40)
    
    
    print(time.perf_counter() - st)