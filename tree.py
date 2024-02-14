from itertools import product

import numpy as np
rng = np.random.default_rng()
from tqdm.notebook import tqdm, trange
from tqdm.contrib.concurrent import process_map

import pandas as pd

from helpers import *
    

def eval_training(modeli, student, teachers, save=False, pct=0):
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
                self.model.rewards[batch] = get_r(self.model)
    
    if student == -1:
        model2 = make_model()
    else:
        model2 = tf.keras.models.load_model(f"models/2x{ms}/starting_{student:.3f}_{modeli%10}.keras")
        
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    model2.rewards = {}
    
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
    model2.rewards[len(X)//bs] = get_r(model2)
        
    with open('/storage1/fs1/chien-ju.ho/Active/gym/tree4.txt', 'a') as f:
        for k, v in model2.rewards.items():
            print(f"{student};{teachers};{pct};{modeli};{k};{v}", file=f, flush=True)

    return (modeli, student, teachers, model2.rewards)

def exp2(mode, n, nb=100, student=None, algo=None):
    modeli = list(range(n))
    students = [-1, 100]
    saves = [False]
    pcts = [0]
    
    if mode == 'one':
        teachers = [[(t, nb*2)] for t in [0.01, 0.032, 0.1, 0.316, 1., 3.162, 10.]]
        saves = [True]
    
    elif mode == 'two-A':
        teachers = (
            [[((0.01,0.032,0.1,0.316), i)] for i in range(4, nb*4+1, 4)]
            + [[((0.316,0.1), i//2), ((0.032,0.01), i//2)] for i in range(4, nb*4+1, 4)]
            + [[(0.316, i//4), (0.1, i//4), (0.032, i//4), (0.01, i//4)] for i in range(4, nb*4+1, 4)]
        )
        
    
    elif mode == 'two-B':
        teachers = [[((0.316,0.1), i//2), ((0.032,0.01), i//2)] for i in range(8, nb*4+1, 8)]
        pcts = [0.25, 0.5, 0.75, 1]
        
    elif mode.startswith('split'):
        k = int(mode[-1])
        teachers = [[(0.1, i*k), (0.01, i)] for i in range(5, nb//2+1, 5)]
        
    elif mode.startswith('optimal'):
        k = int(mode[-1])
        teachers = []
        for budget in range(10, nb+1, 10):
            for split in range(0, budget+1, 5):
                teachers.append([(0.1, k*split), (0.01, (budget-split))])
        
    elif mode.startswith('algo'):
        k = int(mode[-1])
        students = [student]
        n0, n2 = algo
        teachers = [
            [(0.1, n0*k), (0.1, 5*k), (0.01, n2)],
            [(0.1, n0*k), (0.01, n2), (0.01, 5)]
        ]
        
    
    combo = list(product(modeli, students, teachers, saves, pcts))
    # print(combo[::10])
    # print(teachers)
    return process_map(eval_training, *zip(*combo), chunksize=1, ncols=80)


def exp_algo(mode, student, n):
    algo = [5, 5]
    path = []
    while np.sum(algo) < 100:
        print(algo)
        res = exp2(mode, n, nb=100, student=student, algo=algo)
        res = pd.DataFrame(res, columns=['i','student','teacher','reward'])
        res['reward'] = res.reward.apply(lambda x: list(x.values())[0])
        res['teacher'] = res.teacher.apply(tuple)
        res = res.groupby('teacher').reward.mean()
        res = res.sort_index(ascending=False)
        algo[res.values.argmax()] += 5
        path.append([[algo.copy()], res.values.max()])
    return path


if __name__ == '__main__':
    
    from config import *

    ntotal = (2**np.array(ns[1:13])).sum()
    currs = process_map(make_data, students, [ntotal]*len(students), max_workers=17, chunksize=1, ncols=80)
    currs = dict(zip([f"{s:.3f}" for s in students], currs))

    # args = []
    # for pct in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    #     args.append(['two-2', pct])
    # for mode in [
    #     'one',
    #     'two-1',
    #     'three-1','three-2','three-3',
    #     'optimal-1','optimal-2','optimal-3',
    # ]:
    #     args.append([mode, 0])
    # for mode, pct in args:
    #     _ = exp(-1, n=200, nb=100, mode=mode, pct=pct)
        
    for mode in [
        'one',
        'two-A',
        'two-B',
        'split-1','split-2',
        'optimal-1','optimal-2',
    ]:
        print(mode)
        exp2(mode, 150)
        print()
        
    
    # for mode in [
    #     'algo-1',
    #     'algo-2',
    # ]:
    #     path = exp_algo(mode, 100, 50)
    #     print(path)
    #     print()
