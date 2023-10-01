import numpy as np
import gzip
import pickle
from constants import HAZARD_P, SLIP_P, GROUP_SIZE
from tqdm.contrib.concurrent import process_map
    
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def eval_model(map_size, teacher):
    model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{teacher}.keras')
    
    x_test, y_test = [], []

    for group in range(100):
        # print(f'Testing at {map_size} {teacher} {group}', end='\r')
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{group}.npz')
        x, y = data['x'], data['y']
        x_test.append(x)
        y_test.append(y)
    # print()
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    acc = model.evaluate(x_test, y_test, verbose=0)[1]
    with open('output/eval_acc.txt', 'a') as f:
        print(f'{map_size},{teacher},{acc}', file=f, flush=True)
    return
    
if __name__ == '__main__':
    import sys
    map_size = int(sys.argv[1])
    teacher = int(sys.argv[2])
    eval_model(map_size, teacher)
   
    