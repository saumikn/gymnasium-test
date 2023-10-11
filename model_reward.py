from functools import cache
import numpy as np
from mdp import MDPEnv, is_valid
from constants import HAZARD_P, SLIP_P, GROUP_SIZE

def make_model(map_size, num_dense=4):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    lin = [tf.keras.layers.Flatten(input_shape=(4, map_size, map_size))]
    ldenses = [tf.keras.layers.Dense(128, activation='relu') for _ in range(num_dense)]
    lout = [tf.keras.layers.Dense(4)]    
    model = tf.keras.models.Sequential(lin + ldenses + lout)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def model_reward(map_size, model, state, max_iters=100000, gamma=1, verbose=False):
        
    env = MDPEnv(map_size=map_size, hazard_p=HAZARD_P, slip_p=SLIP_P)
    env.reset(state)

    agents = []
    grids = []
    states = []
    target, hazards, rand = env.state['target'], env.state['hazards'], env.state['rand']
    hazard_grid = env.get_grid()[2]
    rand_grid = env.get_grid()[3]
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
    
    
    if isinstance(model, int):
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.get_logger().setLevel('ERROR')  
        model = tf.keras.models.load_model(f'/storage1/fs1/chien-ju.ho/Active/gym/models{map_size}/myopic_{model}.keras')
        actions = model.predict(np.array(grids)).argmax(axis=1)
    elif isinstance(model, np.ndarray):
        actions = model
    else: # Is keras model already
        actions = model.predict(np.array(grids)).argmax(axis=1)
        
    trans = dict(zip(agents, actions))
    visited = {}

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
    
    values = np.zeros((map_size,map_size))
    for vi in range(max_iters):
        # print(vi, end='\r')
        next_values = np.zeros_like(values)
        for i, j in agents:
            for prob, (i2,j2), reward in next_agents((i,j)):
                # print(prob, (i,j), (i2,j2), reward)
                # next_values[i][j] += prob * (gamma*values[i2,j2] + reward)
                next_values[i][j] += prob * (gamma*values[i2,j2] + reward - rand_grid[i2,j2])
        if np.abs(values-next_values).sum() < 0.0001:
            break
        values = next_values
    # print()
        
    avg_value = np.mean([values[i,j] for (i,j) in agents])
        
    if verbose == False:
        return np.mean([values[i,j] for (i,j) in agents])
    
    dirs = np.zeros((map_size, map_size), dtype=str)
    dirs[:] = ' '
    key = {0:'↓', 1:'→', 2:'↑', 3:'←'}
    for i, j in agents:
        dirs[i,j] = key[trans[(i,j)]]
        
        
    val_reward = values
    val_reward += 10*env.get_grid()[1]
    val_reward += -1*env.get_grid()[2]
    val_reward += env.get_grid()[3]
    
    return avg_value, dirs, val_reward
