import numpy as np
rng = np.random.default_rng()

from tqdm.notebook import tqdm, trange
from tqdm.contrib.concurrent import process_map

from config import *


def make_model(num_nodes=32, num_dense=3):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    inputs = tf.keras.layers.Input(shape=(ms*ms + ms + 1))
    x = tf.keras.layers.Flatten()(inputs)
    for _ in range(num_dense):
        x = tf.keras.layers.Dense(num_nodes, activation='relu')(x)
    output1 = tf.keras.layers.Dense(ms, name='Y0')(x)
    output1 = tf.keras.layers.Softmax()(output1)
    model = tf.keras.models.Model(inputs=inputs, outputs=output1)
    
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    return model

def softmax(x, b, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x/b) / np.exp(x/b).sum(axis=axis, keepdims=True)

def make_x():
    x = arrs
    # rand = np.random.random(arrs.shape)*0.4 - 0.2
    # x = x + rand
    x = rng.permuted(x, axis=-1)
    return x

def eval_model(model, n = 1, disable=True):
    rewards = []
    branch_rewards = []
    for _ in trange(n, disable=disable):
        x = make_x()
        x_tree = np.concatenate([np.eye(ms+1).astype(int), np.tile(x.flatten(), (ms+1,1))], axis=1)
        y_pred = model(x_tree).numpy()
        y_pred[0,2:] = 0
        y_pred[0] = y_pred[0] / y_pred[0].sum()
        branch_reward = (x_tree[0,ms+1:].reshape(ms,ms) * y_pred[1:]).sum(axis=1)
        rewards.append(y_pred[0])
        branch_rewards.append(branch_reward)
    return np.array(rewards), np.array(branch_rewards)

def make_data(b=1, n=1, disable=True):
    X, Y = [], []
    for _ in trange(n, disable=disable):
        
        x = make_x()
        r1 = softmax(x, b)
        r1sum = (r1*x).sum(axis=1)
        r0 = softmax(r1sum, b)
        
        pos = [0] * (ms+1)
        pos[0] = 1
        x0 = pos + list(x.flatten())
                
        slip_prob = b
        
        if slip == False:
            p = np.zeros_like(r0)
            p[:2] = r0[:2]
            p = p / p.sum()
            y0 = rng.choice(np.arange(ms), p=p)
        else:
            p = np.zeros_like(r0)
            p[0] = 1-slip_prob
            p[1:] = slip_prob/(ms-1)
            y0 = rng.choice(np.arange(ms), p=p)            
        
        pos = [0] * (ms+1)
        pos[y0+1] = 1
        x1 = pos + list(x.flatten())
        
        if slip == False:
            y1 = rng.choice(np.arange(ms), p=r1[y0])
        else:
            p = np.zeros(ms) + (slip_prob)/(ms-1)
            p[r1.argmax()] = 1 - slip_prob
            y1 = rng.choice(np.arange(ms), p=p)

        X.append(x0)
        Y.append(y0)
        X.append(x1)
        Y.append(y1)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X,Y
