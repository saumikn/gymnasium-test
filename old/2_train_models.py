import numpy as np
import gzip
import pickle
from constants import HAZARD_P, SLIP_P, GROUP_SIZE
from tqdm.contrib.concurrent import process_map
from model_reward import make_model
    
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def train_model(map_size, teacher):
    model = make_model(map_size)
    
    for group in range(600):
        print(f'Training data at {map_size} {teacher} {group}', flush=True)
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/train/myopic_{teacher}_{group}.npz')
        x = data['x']
        y = data['y']
        model.fit(x, y, verbose=0)
    print()
    # model.save(f'tmp/myopic_{map_size}_{teacher}.keras')
    model.save(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{teacher}.keras')
    return

# def eval_model(map_size, teacher):
#     model = make_model(map_size)
#     model = tf.keras.models.load_model(f'tmp/myopic_{map_size}_{teacher}.keras')
    
#     x_test, y_test = [], []

#     for group in range(100):
#         print(f'Testing at {map_size} {teacher} {group}', end='\r')
#         data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{group}.npz')
#         x, y = data['x'], data['y']
#         x_test.append(x)
#         y_test.append(y)
#     x_test = np.concatenate(x_test)
#     y_test = np.concatenate(y_test)
#     print(x_test.shape, y_test.shape)
#     model.evaluate(x_test, y_test)
#     return
    
if __name__ == '__main__':
    import sys
    map_size = int(sys.argv[1])
    teacher = int(sys.argv[2])
    border = int(sys.argv[2])
    
    train_model(map_size, teacher)
    # eval_model(map_size, teacher)
   
    