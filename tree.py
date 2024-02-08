from collections import Counter
from functools import partial

import numpy as np
rng = np.random.default_rng()
np.set_printoptions(suppress=True, linewidth=180, edgeitems=5)
from tqdm.notebook import tqdm, trange
from tqdm.contrib.concurrent import process_map

from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings


def make_model(num_nodes=128, num_dense=4):
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
    
    # rand = np.random.random(arrs.shape)
    # rand = (rand - 0.5)
    # x = x + rand*0.0001
    
    x = rng.permuted(x, axis=-1)
    return x

def eval_model(model, n = 1, disable=True):
    rewards = []
    branch_rewards = []
    for _ in trange(n, disable=disable):
        x = make_x()
        x_tree = np.concatenate([np.eye(ms+1).astype(int), np.tile(x.flatten(), (ms+1,1))], axis=1)
        y_pred = model(x_tree).numpy()
        
        if top2:
            y_pred[0,2:] = 0
            y_pred[0] = y_pred[0] / y_pred[0].sum()
        
        branch_reward = (x_tree[0,ms+1:].reshape(ms,ms) * y_pred[1:]).sum(axis=1)
        rewards.append(y_pred[0])
        # reward = (y_pred[0] * branch_reward).numpy().sum()
        # rewards.append(reward)
        branch_rewards.append(branch_reward)
    return np.array(rewards), np.array(branch_rewards)

def make_data(b=1, n=1, disable=True):
    X, Y = [], []
    for _ in trange(n, disable=disable):
        
        if ranked:
            x = make_x()
            r1 = softmax(rankdata(x,axis=1), b)
            r1sum = (r1*x).sum(axis=1)
            r0 = softmax(rankdata(r1sum), b)
                    
        else:
            x = make_x()
            r1 = softmax(x, b)
            r1sum = (r1*x).sum(axis=1)
            r0 = softmax(r1sum, b)
        
        pos = [0] * (ms+1)
        pos[0] = 1
        x0 = pos + list(x.flatten())
        
        
        # slip_prob = np.log10(b)/-20 + 0.95
        slip_prob = b
        
        if slip == False:
            if top2:
                p = np.zeros_like(r0)
                p[:2] = r0[:2]
                p = p / p.sum()
            else:
                p = r0
            y0 = rng.choice(np.arange(ms), p=p)
        else:
            if top2:
                p = np.zeros_like(r0)
                p[0:2] = [1-slip_prob, slip_prob]
            else:
                p = np.zeros_like(r0)
                p[0] = 1-slip_prob
                p[1:] = slip_prob/(ms-1)
            y0 = rng.choice(np.arange(ms), p=p)
        # while y0 in [2,3,4]:
        #     y0 = rng.choice(np.arange(ms), p=r0)
            
        
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
    

def eval_training(modeli, student, teachers, save, pct=0):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')
    
    def get_r(model):
        r0, r1 = eval_model(model, 10)
        r = (r0*r1).sum(axis=1).mean()
        return r
        
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(CustomCallback, self).__init__()
        def on_batch_begin(self, batch, logs=None):            
            if save and batch % 1 == 0:
                r = get_r(self.model)
                try:
                    self.model.rewards[batch] = r
                except:
                    self.model.rewards = {batch: r}
    
    if student == -1:
        model2 = make_model()
    else:
        model2 = tf.keras.models.load_model(f"models/2x{ms}/starting_{student:.3f}_{modeli%10}.keras")
        
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    
    bs = 32
    
    rng = np.random.default_rng(modeli)
    
    X, Y = [], []
    for teacher, n in teachers:
        if isinstance(teacher, tuple):
            X.append([])
            Y.append([])
            for teacher_part in teacher:
                _X, _Y = currs[f"{teacher_part:.3f}"]
                perm = np.random.default_rng(modeli).permutation(len(_X))[:bs * n//len(teacher)]
                # perm = np.random.permutation(len(_X))[:bs * n//len(teacher)]
                X[-1].append(_X[perm])
                Y[-1].append(_Y[perm])
            X[-1] = np.concatenate(X[-1])
            Y[-1] = np.concatenate(Y[-1])
            perm = np.random.permutation(len(X[-1]))
            X[-1] = X[-1][perm]
            Y[-1] = Y[-1][perm]
        else:
            _X, _Y = currs[f"{teacher:.3f}"]
            perm = np.random.permutation(len(_X))[:n*bs]
            X.append(_X[perm])
            Y.append(_Y[perm])

    lx = len(X[0])
    if pct:
        assert (pct*lx) % 1 == 0
        pct2 = int(pct * lx)
        mixX = np.concatenate([_X[lx-pct2:] for _X in X])
        mixY = np.concatenate([_Y[lx-pct2:] for _Y in Y])
        for i, (_X, _Y) in enumerate(zip(X, Y)):
            _X[lx-pct2:] = mixX[i::len(X)]
            _Y[lx-pct2:] = mixY[i::len(Y)]
            perm = np.random.permutation(lx)
            _X[:] = _X[perm]
            _Y[:] = _Y[perm]
    
    X, Y = np.concatenate(X), np.concatenate(Y)        
    model2.fit(X, Y, verbose=False, shuffle=False, batch_size=bs, callbacks=[CustomCallback()]) 
    
    final_reward = get_r(model2)
    try:
        model2.rewards[len(X)//bs] = final_reward
    except:
        model2.rewards = {len(X)//bs: final_reward}
        
        
    with open('/storage1/fs1/chien-ju.ho/Active/gym/tree.txt', 'a') as f:
        for k, v in model2.rewards.items():
            print(f"{student};{teachers};{pct};{modeli};{k};{v}", file=f, flush=True)

    return (modeli, student, teachers, model2.rewards)


# def exp(student, verbose=True, n=20, nb=100, mode='one', algo=None, pct=0):
        
#     if mode == 'one':
#         teachers = [
#             [(0.010, nb)],
#             [(0.032, nb)],
#             [(0.100, nb*3)],
#             [(0.316, nb)],
#             [(1.000, nb)],
#             [(3.162, nb)],
#             [(10.000, nb)],
#         ]
        
#     elif mode == 'two-1':
#         teachers = [
#             [((0.010,0.100,1.000,10.00), nb)],
#             [((10.00,1.000), nb//2), ((0.100,0.010), nb//2)],
#             [(10.00, nb//4), (1.000, nb//4), (0.100, nb//4), (0.010, nb//4)],
#         ]
#     elif mode == 'two-2':
#         teachers = [
#             [((10.00,1.000), nb//2), ((0.100,0.010), nb//2)],
#         ]
        
#     elif mode == 'three-1':
#         teachers = [[(0.1, i), (0.01, i)] for i in range(1, nb//2+1)]
#     elif mode == 'three-2':
#         teachers = [[(0.1, i*2), (0.01, i)] for i in range(1, nb//2+1)]
#     elif mode == 'three-3':
#         teachers = [[(0.1, i*3), (0.01, i)] for i in range(1, nb//2+1)]
        
            
#     elif mode == 'algo-1':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.1*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0 + [(0.1, nb*0.1)] + t1)
#                 teachers.append(t0 + t1 + [(0.01, nb*0.1)])
#     elif mode == 'algo-2':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.2*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0 + [(0.1, nb*0.2)] + t1)
#                 teachers.append(t0 + t1 + [(0.01, nb*0.1)])
#                 teachers.append(t)        
#     elif mode == 'algo-3':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.3*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0 + [(0.1, nb*0.3)] + t1)
#                 teachers.append(t0 + t1 + [(0.01, nb*0.1)])
                
#     elif mode == 'optimal-1':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.1*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0+t1)
#     elif mode == 'optimal-2':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.2*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0+t1)
#     elif mode == 'optimal-3':
#         teachers = []
#         for i in range(1, 11):
#             for split in range(1, i):
#                 t0, t1 = [(0.1, nb*0.3*split)], [(0.01, nb*0.1*(i-split))]
#                 teachers.append(t0+t1)

#     t_, s_, iters, saves, pcts = [], [], [], [], []
#     for teacher in teachers:
#         for i in range(n):
#             t_.append([(t1, int(t2)) for t1, t2 in teacher])
#             s_.append(student)
#             iters.append(i)
#             if mode.startswith('three') or mode.startswith('optimal') or mode.startswith('algo'):
#                 saves.append(False)
#             else:
#                 saves.append(True)
                
#             pcts.append(pct)
        
#     res = process_map(eval_training, iters, s_, t_, saves, pcts, max_workers=21, chunksize=1)
#     return res

def exp(student, verbose=True, n=20, nb=100, mode='one', algo=None, pct=0):
        
    if mode == 'one':
        teachers = [
            [(0.010, nb)],
            [(0.032, nb)],
            [(0.100, nb*3)],
            [(0.316, nb)],
            [(1.000, nb)],
            [(3.162, nb)],
            [(10.000, nb)],
        ]
        
    elif mode == 'two-1':
        teachers = [
            [((0.010,0.100,1.000,10.00), nb)],
            [((10.00,1.000), nb//2), ((0.100,0.010), nb//2)],
            [(10.00, nb//4), (1.000, nb//4), (0.100, nb//4), (0.010, nb//4)],
        ]
    elif mode == 'two-2':
        teachers = [
            [((10.00,1.000), nb//2), ((0.100,0.010), nb//2)],
        ]
        
    elif mode.startswith('three'):
        k = int(mode[-1])
        teachers = [[(0.1, i*k), (0.01, i)] for i in range(1, nb//2+1)]
                                
    elif mode.startswith('algo'):
        k = int(mode[-1])
        n0, n2 = algo
        teachers = []
        teachers.append([(0.1, n0*k), (0.1, 10*k), (0.01, n2)])
        teachers.append([(0.1, n0*k), (0.01, n2), (0.01, 10)])
        
    elif mode.starswith('optimal'):
        k = int(mode[-1])
        teachers = []
        for i in range(1, 11):
            for split in range(1, i):
                t0, t1 = [(0.1, nb*0.1*k*split)], [(0.01, nb*0.1*(i-split))]
                teachers.append(t0+t1)
    
    t_, s_, iters, saves, pcts = [], [], [], [], []
    for teacher in teachers:
        for i in range(n):
            t_.append([(t1, int(t2)) for t1, t2 in teacher])
            s_.append(student)
            iters.append(i)
            if mode.startswith('three') or mode.startswith('optimal') or mode.startswith('algo'):
                saves.append(False)
            else:
                saves.append(True)
                
            pcts.append(pct)
        
    res = process_map(eval_training, iters, s_, t_, saves, pcts, max_workers=21, chunksize=1)
    return res

def exp_algo(mode):
    path = []
    algo = [10, 10]
    while np.sum(algo) < 100:
        print(algo)
        res = exp(100, n=1000, nb=100, mode=mode, pct=0, algo=algo)
        res = pd.DataFrame(res, columns=['i','student','teacher','reward'])
        res['reward'] = res.reward.apply(lambda x: list(x.values())[0])
        res['teacher'] = res.teacher.apply(tuple)
        res = res.groupby('teacher').reward.mean()
        res = res.sort_index(ascending=False)
        path.append([[algo.copy()], res.values.max()])
        algo[res.values.argmax()] += 10
    return path


if __name__ == '__main__':
    
    
    ms = 2

    if ms == 10:
        students = np.logspace(-2, 1, 13)
        ns = [0,5,5,6,7,8,9,10,11,12,13,14,15]
        arrs = np.array([[1,0,0,0,0,0,0,0,0,0],
                         [.95,.9,.85,.8,.75,.7,.65,.6,.55,.5],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                        ])

    elif ms == 2:
        students = np.logspace(-2, 2, 17)
        ns = [0,5,5,6,7,8,9,10,11,12,13,14,15]
        arrs = np.array([[1,0],[.95,.90]])

    ranked = False
    top2 = True
    slip = False

    
    ntotal = (2**np.array(ns[1:13])).sum()
    currs = process_map(make_data, students, [ntotal]*len(students), max_workers=17, chunksize=1)
    currs = dict(zip([f"{s:.3f}" for s in students], currs))

    # args = []
    # for pct in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    #     args.append(['two-2', pct])
#     for mode in [
#         'one',
#         # 'two-1',
#         # 'three-1','three-2','three-3',
#         # 'optimal-1','optimal-2','optimal-3',
#         # 'algo-1','algo-2','algo-3'
#     ]:
#         args.append([mode, 0])
#     for mode, pct in args:
#         _ = exp(100, n=1000, nb=100, mode=mode, pct=pct)
        
    
    for mode in ['algo-1', 'algo-2', 'algo-3']:
        path = exp_algo(mode)
        print(path)
        print()
