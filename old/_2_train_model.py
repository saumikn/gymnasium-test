from tqdm.contrib.concurrent import process_map
import numpy as np
from constants import BS, DIR
import os
from pathlib import Path
from itertools import product
from scipy.spatial.distance import cdist

save_points = [2000,4000,8000,16000,32000,
               64000,128000,256000,512000,
               1024000,2048000,4096000]

def train_model(map_size, student, teacher, train_size, seed, num_dense, num_nodes, markers):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
        
    def make_model(map_size):
        channels = 4
        if markers:
            channels += 2*map_size - 1
        lin = [tf.keras.layers.Flatten(input_shape=(channels, map_size, map_size))]
        ldenses = [tf.keras.layers.Dense(num_nodes, activation='relu') for _ in range(num_dense)]
        lout = [tf.keras.layers.Dense(4)]    
        model = tf.keras.models.Sequential(lin + ldenses + lout)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        return model

    
    Path(f'{DIR}/models_{num_dense}_{num_nodes}_{map_size}_{markers}').mkdir(parents=True, exist_ok=True)
        
    if os.path.isfile(f'{DIR}/models_{num_dense}_{num_nodes}_{map_size}_{markers}/{student}_{teacher}_{train_size}_{seed}.keras'):
        return
    
    if student:
        model = tf.keras.models.load_model(f'{DIR}/models_{num_dense}_{num_nodes}_{map_size}_{markers}/{0}_{student}_{2048000}_{seed%1}.keras')
    else:
        model = make_model(map_size)

    rng = np.random.default_rng(seed)

    bpg = 2000 # batches per group
    
    train_groups = list(rng.choice(1024, size=1024, replace=False))
    train_groups += train_groups
    for i, group in enumerate(train_groups):
        if i*bpg >= train_size:
            break
        print(f'Training at {i}/{len(train_groups)}     ', end='\r')
        data = np.load(f'{DIR}/data{map_size}/train/myopic_{teacher}_{group}.npz')
        with tf.device("CPU"):
            x = data['x']
            markers_arr = np.zeros((len(x), map_size*2-1, map_size, map_size))
            if markers:
                for j in range(len(x)):
                    xj = x[j]
                    mj = markers_arr[j]
                    xa = np.argwhere(xj[0])
                    xb = np.array(list(product(*[range(k) for k in xj[0].shape])))
                    dists = cdist(xa, xb, metric='cityblock')
                    idxs = np.hstack([dists.T, xb]).astype(int)
                    mj[tuple(idxs.T)] = 1
                x = np.concatenate((x, markers_arr), axis=1)

            train_dataset = tf.data.Dataset.from_tensor_slices((x, data['y']))
            train_dataset = train_dataset.shuffle(100000, seed=seed).batch(BS).take(bpg)
        model.fit(train_dataset, verbose=2)
        ds = (i+1)*bpg
        if ds in save_points:
            model.save(f'{DIR}/models_{num_dense}_{num_nodes}_{map_size}_{markers}/{student}_{teacher}_{ds}_{seed}.keras')


if __name__=='__main__':

    import sys
    map_size = int(sys.argv[1])
    student = int(sys.argv[2])
    teacher = int(sys.argv[3])
    train_size = int(sys.argv[4])
    num_dense = int(sys.argv[5])
    num_nodes = int(sys.argv[6])
    offset = int(sys.argv[7])
    markers = int(sys.argv[8])
    
    simul = 5
        
    iterables = [(map_size,student,teacher,train_size,seed,num_dense,num_nodes,markers)
                 for seed in range(offset, offset+simul)]    
    process_map(train_model, *zip(*iterables), chunksize=1, max_workers=5)