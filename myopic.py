import numpy as np
import pandas as pd
from mdp import MDPEnv
from functools import cache, lru_cache
from collections import defaultdict

MAP_SIZE = 3  # Number of tiles of one side of the squared environment
HAZARD_P = 0.5  # Probability that square is hazadorous
SLIP_P = 0.4  # Probability of slipping on the ice


def exp(seed):
    @cache
    def myopic(state, k=1):
        if k == 0:
            return None, 0

        env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P)
        results = defaultdict(list)
        for _ in range(100):
            env.reset(state)
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                results[action].append(reward)
            else:
                _, next_reward = myopic(env.get_state(), k - 1)
                results[action].append(reward + next_reward)
        env.reset(state)

        results = {k: np.mean(v) for k, v in results.items()}
        best_action = max(results, key=results.get)
        best_reward = results[best_action]
        return best_action, best_reward

    env = MDPEnv(map_size=MAP_SIZE, hazard_p=HAZARD_P, slip_p=SLIP_P, seed=seed)
    env.reset()
    state = env.get_state()
    best_action, _ = myopic(state, k=3)
    return env.get_grid(), best_action
