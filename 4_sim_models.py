from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
import numpy as np
from model_reward import model_reward

if __name__ == '__main__':
    import sys
    map_size = int(sys.argv[1])
    teacher = int(sys.argv[2])
    
    x = []
    for group in range(10):
        data = np.load(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{teacher}_{group}.npz')
        x.append(data['x'])
    x = np.concatenate(x)
    
    x = x[::map_size**2]
    x = x[:1000]
        
    map_sizes = [map_size for _ in x]
    teachers = [teacher for _ in x]
    
    res = process_map(model_reward, map_sizes, teachers, x, disable=True)
    
    with open('output/eval_sim.txt', 'a') as f:
        print(f'{map_size},{teacher},{np.mean(res)}', file=f, flush=True)
