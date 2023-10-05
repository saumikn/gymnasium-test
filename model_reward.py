from functools import cache
import numpy as np
from mdp import MDPEnv, is_valid
from constants import HAZARD_P, SLIP_P, GROUP_SIZE


def model_reward(model, map_size, state):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel('ERROR')  
    
    
    if isinstance(model, int):
        model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{model}.keras')
    
    env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
    env.reset(state)

    agents = []
    grids = []
    states = []
    target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
    hazard_grid = env.get_grid()[2]
    for i in range(map_size):
        for j in range(map_size):
            if (i,j) == target or (i,j) in hazards:
                continue
            if not is_valid(hazard_grid, (i,j), target):
                continue
            state = ((i,j), target, hazards, rand)
            env.reset(state)
            agents.append((i,j))
            grids.append(env.get_grid())
            states.append(state)
    
    actions = model.predict(np.array(grids)).argmax(axis=1)
    trans = dict(zip(agents, actions))
    visited = {}
    values = np.zeros((map_size,map_size))

    @cache
    def next_agents(agent):
        env.reset(state)
        env.state['agent'] = agent
        action = trans[agent]
        probs, next_agents = env.next_agents(action)
        res = []
        for prob, next_agent in zip(probs, next_agents):
            env.reset(state)
            env.state['agent'] = agent
            _, reward, terminated, truncated, _ = env.step(next_agent=next_agent)
            res.append((prob, next_agent, reward))
        return res

    for vi in range(100):
        next_values = np.zeros_like(values)
        for i, j in agents:
            for prob, (i2,j2), reward in next_agents((i,j)):
                # print(prob, (i,j), (i2,j2), reward)
                next_values[i][j] += prob * (values[i2,j2] + reward)
        values = next_values

    return np.mean([values[i,j] for (i,j) in agents])
