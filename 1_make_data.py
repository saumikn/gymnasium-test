from mdp import MDPEnv, is_valid
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm, trange
import pickle
import gzip
from functools import cache
import numpy as np
import gc

from constants import HAZARD_P, SLIP_P, GROUP_SIZE


def max_model(map_size):
    return int(map_size*2.5+1)

def exp(map_size, seed):
    
    rng = np.random.default_rng(seed)
    
    @cache
    def myopic(state, k=1):
        if k == 0: return None, 0
        env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
        results = []
        actions = list(range(env.action_space.n))
        rng.shuffle(actions)
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
    
    env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P, seed=seed)
    state, _ = env.reset()
    target = state['target']
    hazards = state['hazards']
    rand = state['rand']
    
    results = []
    
    hazard_grid = env.get_grid()[2]
    
    for m in range(1, max_model(map_size)):
        for i in range(map_size):
            for j in range(map_size):
                if (i,j) == target or (i,j) in hazards:
                    continue
                if not is_valid(hazard_grid, (i,j), target):
                    continue
                state = ((i,j), target, hazards, rand)
                env.reset(state)
                
                best_action, best_reward = myopic(state, k=m)
                results.append((m, env.get_grid(), best_action, best_reward))
                
    return results



if __name__ == '__main__':
    import sys
    map_size = int(sys.argv[1])
    group = int(sys.argv[2])
    sizes = [map_size for _ in range(GROUP_SIZE)]
    myopics = list(range(int(map_size*2.5+1)))
    
    disable=True
    
    groups = list(range(group*GROUP_SIZE, group*GROUP_SIZE + GROUP_SIZE))
    res = process_map(exp, sizes, groups, chunksize=1, disable=disable)
    res = [j for i in res for j in i]
    for myo in trange(1, max_model(map_size), disable=disable):
        res_myo = [i for i in res if i[0]==myo]
        grids, actions = zip(*[(g,a) for _,g,a,_ in res_myo])
        grids, actions = np.array(grids), np.array(actions)
        split = int(len(grids)*0.8)
        x_train, x_test = grids[:split], grids[split:]
        y_train, y_test = actions[:split], actions[split:]

        np.savez_compressed(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/train/myopic_{myo}_{group}.npz', x=x_train, y=y_train)
        np.savez_compressed(f'/storage1/fs1/chien-ju.ho/Active/gym/data{map_size}/test/myopic_{myo}_{group}.npz', x=x_test, y=y_test)