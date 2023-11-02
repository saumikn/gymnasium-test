import numpy as np

def flat(l):
    return [j for i in l for j in i]

def batch(iterable, n = 100, enum=False):
    '''Helper function which turns iterable into batches of size n. Last batch may be smaller than n'''
    i = 0
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            if enum:
                yield (i, current_batch)
            else:
                yield current_batch
            i += 1
            current_batch = []
    if current_batch:
        if enum:
            yield (i, current_batch)
        else:
            yield current_batch

def batchl(iterable, n=100, enum=False):
    return list(batch(iterable, n, enum))

def sortd(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

def sem(a):
    a = np.array(a)
    return a.std() / np.sqrt(len(a))