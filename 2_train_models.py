# import tensorflow as tf
# for gpu in tf.config.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import gzip
import pickle
from constants import MYOPIC, MAP_SIZE
from tqdm.contrib.concurrent import process_map
    
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def make_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(4, MAP_SIZE, MAP_SIZE)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def train_model(myo):
    filename = f'/storage1/fs1/chien-ju.ho/Active/gym/data/myopic_{myo}.gzip'
    with gzip.GzipFile(filename, 'rb') as f:
        res = pickle.loads(f.read())
    grids, actions = zip(*[(g,a) for _,g,a,_ in res])
    grids, actions = np.array(grids), np.array(actions)
    print(f'training myopic = {myo}, n={len(grids)}')
    split = int(len(grids)*0.8)
    x_train, x_test = grids[:split], grids[split:]
    y_train, y_test = actions[:split], actions[split:]

    model = make_model()
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)
    print()
    model.save(f'/storage1/fs1/chien-ju.ho/Active/gym/models/myopic_{myo}.keras')
        
for myo in [5,6,7]:
    train_model(myo)
                  
# process_map(train_model, MYOPIC, disable=True, max_workers=3)
    
    