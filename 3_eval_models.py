import numpy as np
import gzip
import pickle
from constants import HAZARD_P, SLIP_P, GROUP_SIZE, DIR, BS
from tqdm.contrib.concurrent import process_map

def eval_model(map_size, teacher, train_size, seed):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{teacher}.keras')
    model = tf.keras.models.load_model(f'{DIR}/models{map_size}/{0}_{teacher}_{train_size}_{seed}.keras')
    
    x_test, y_test = [], []

    rng = np.random.default_rng(seed)
    test_groups = list(rng.choice(1024, size=100, replace=False))
    
    for group in test_groups:
        # print(f'Testing at {map_size} {teacher} {group}', end='\r')
        data = np.load(f'{DIR}/data{map_size}/test/myopic_{teacher}_{group}.npz')
        x, y = data['x'], data['y']
        x_test.append(x)
        y_test.append(y)
    # print()
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    
    with tf.device("CPU"):
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.take(len(x_test)).batch(BS)

    acc = model.evaluate(test_dataset, verbose=0)[1]
    with open('output/eval_acc.txt', 'a') as f:
        print(f'{map_size},{teacher},{train_size},{seed},{acc}', file=f, flush=True)
    return
    
if __name__ == '__main__':
    # import sys
    # map_size = int(sys.argv[1])
    # teacher = int(sys.argv[2])
    # eval_model(map_size, teacher)
    
    iterables = []
    for map_size in [8]:
        for train_size in [1024000]:
            for teacher in range(1,21):
                for seed in range(5):
                    iterables += [(map_size,teacher,train_size,seed)]
                
    process_map(eval_model, *zip(*iterables), max_workers=5)
   
    