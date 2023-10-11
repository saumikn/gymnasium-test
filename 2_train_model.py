from tqdm.contrib.concurrent import process_map
import numpy as np
from constants import BS, DIR
from model_reward import make_model
import gc
import os
from pathlib import Path


save_points = [2000,4000,8000,16000,32000,
               64000,128000,256000,512000,
               1024000,2048000]

def tune_model(map_size, student, teacher, train_size, seed):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    Path(f'{DIR}/models{map_size}').mkdir(parents=True, exist_ok=True)
        
    if os.path.isfile(f'{DIR}/models{map_size}_{student}_{teacher}_{train_size}_{seed}.keras'):
        return
    
    if student:
        model = tf.keras.models.load_model(f'{DIR}/models{map_size}/{0}_{student}_{2048000}_{seed}.keras')
    else:
        model = make_model(map_size, num_dense=6)

    rng = np.random.default_rng(seed)

    bpg = 2000 # batches per group
    
    train_groups = list(rng.choice(1024, size=train_size//bpg, replace=False))
    for i, group in enumerate(train_groups):
        print(f'Training at {i}/{len(train_groups)}     ', end='\r')
        data = np.load(f'{DIR}/data{map_size}/train/myopic_{teacher}_{group}.npz')
        with tf.device("CPU"):
            train_dataset = tf.data.Dataset.from_tensor_slices((data['x'], data['y']))
            train_dataset = train_dataset.shuffle(10000, seed=seed).batch(BS).take(bpg)
        model.fit(train_dataset, verbose=0)
        ds = (i+1)*bpg
        if ds in save_points:
            model.save(f'{DIR}/models{map_size}/{student}_{teacher}_{ds}_{seed}.keras')


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
        
    iterables = [(map_size,student,teacher,train_size,seed) for seed in range(offset, offset+5)]    
    process_map(tune_model, *zip(*iterables), chunksize=1, max_workers=5)