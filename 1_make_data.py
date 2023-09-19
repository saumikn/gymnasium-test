from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import pickle
import gzip
from functools import cache
import numpy as np

from constants import MAP_SIZE, HAZARD_P, SLIP_P, MYOPIC, SEEDS

def exp(seed):
    @cache
    def myopic(state, k=1):
        if k == 0: return None, 0
        env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P)
        results = []
        actions = list(range(env.action_space.n))
        np.random.shuffle(actions)
        for action in actions:
            env.reset(state)
            probs, next_agents = env.next_agents(action)
            
            reward_sum = 0
            for p, na in zip(probs, next_agents):
                _, reward, terminated, truncated, _ = env.step(next_agent = na)
                if terminated or truncated:
                    reward_sum += p * reward
                else:
                    _, next_reward = myopic(env.get_tuple(), k - 1)
                    reward_sum += p * (reward + next_reward)
            results.append((reward_sum, action))
        best_reward, best_action = max(results)
        return best_action, best_reward    
    
    env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P, seed=seed)
    state, _ = env.reset()
    target = state['target']
    hazards = state['hazards']
    rand = state['rand']
    
    results = []
    
    hazard_grid = env.get_grid()[2]
    
    for m in MYOPIC:
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if (i,j) == target or (i,j) in hazards:
                    continue
                if not is_valid(hazard_grid, (i,j), target):
                    continue
                state = ((i,j), target, hazards, rand)
                env.reset(state)
                
                best_action, best_reward = myopic(state, k=m)
                results.append((m, env.get_grid(), best_action, best_reward))
                
    return results




res = process_map(exp, range(SEEDS), chunksize=1, mininterval=0.5)
res = [j for i in res for j in i]
for myo in tqdm(MYOPIC):
    res_myo = [i for i in res if i[0]==myo]
    filename = f'/storage1/fs1/chien-ju.ho/Active/gym/data/myopic_{myo}.gzip'
    with gzip.GzipFile(filename, 'wb') as f:
        f.write(pickle.dumps(res_myo))
        
        
for myo in MYOPIC:
    filename = f'/storage1/fs1/chien-ju.ho/Active/gym/data/myopic_{myo}.gzip'
    with gzip.GzipFile(filename, 'rb') as f:
        res = pickle.loads(f.read())
        grids, actions = zip(*[(g,a) for _,g,a,_ in res])
        grids, actions = np.array(grids), np.array(actions)
        split = int(len(grids)*0.8)
        x_train, x_test = grids[:split], grids[split:]
        y_train, y_test = actions[:split], actions[split:]
        
        np.savez_compressed(f'/storage1/fs1/chien-ju.ho/Active/gym/data/train/myopic_{myo}.npz', x=x_train, y=y_train)
        np.savez_compressed(f'/storage1/fs1/chien-ju.ho/Active/gym/data/test/myopic_{myo}.npz', x=x_test, y=y_test)