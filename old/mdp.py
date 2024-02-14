import numpy as np
import gymnasium as gym
from gymnasium import spaces
from constants import GOAL, HAZARD, RAND


def is_valid(grid, agent, target):
    """Checks if a randomly initialized grid is valid, meaning there is a continuous path from start to finish."""
    rows, cols = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] or visited[r][c]:
            return False
        if (r, c) == target:
            return True
        visited[r][c] = True
        return dfs(r + 1, c) or dfs(r - 1, c) or dfs(r, c + 1) or dfs(r, c - 1)

    return dfs(*agent)


def generate_random_map(size=8, p=0.8, seed=None):
    """Generates a random valid map (one that has a path from start to target)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """

    rng = np.random.default_rng(seed)

    grid = np.zeros((4, size, size))
    p = min(p, 1)

    # valid = False
    # while not valid:
    a, t = rng.choice(grid[0].size, size=2, replace=False)
    a = (a // size, a % size)
    t = (t // size, t % size)
    grid[0, a[0], a[1]] = 1
    grid[1, t[0], t[1]] = 1
    grid[2] = rng.choice([0, 1], size=(size, size), p=[1 - p, p])
    grid[2, a[0], a[1]] = 0  # Agent isn't on a hazard
    grid[2, t[0], t[1]] = 0  # Target isn't on a hazard
    # valid = is_valid(grid[2], a, t)
    hazards = frozenset(map(tuple, np.argwhere(grid[2])))
    grid[3] = rng.uniform(-RAND, 0, (size, size))
    rand = tuple(tuple(i) for i in grid[3])

    return grid, {'agent':a, 'target':t, 'hazards':hazards, 'rand':rand}


class MDPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_size=4, hazard_p=0.8, slip_p=0, seed=None):
        self.render_mode = None
        self.map_size = map_size
        self.hazard_p = hazard_p
        self.slip_p = slip_p
        self.action_size = 4  # Hardcoded to Up, Down, Left, Right
        self.seed = seed

        """
        The obversation space is a n by n grid.
        The first channel represents the location of the agent.
        The second channel represents the location of the target.
        The third channel represents the location of the hazards.
        """
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.map_size, self.map_size), dtype=int
        )

        self.action_space = spaces.Discrete(self.action_size)
        self._action_to_direction = [
            np.array([1, 0]),  # Down
            np.array([0, 1]),  # Right
            np.array([-1, 0]),  # Up
            np.array([0, -1]),  # Left
        ]

    def get_state(self):
        return self.state
    
    def get_tuple(self):
        return (
            self.state['agent'],
            self.state['target'],
            self.state['hazards'],
            self.state['rand'],
        )

    def get_grid(self):
        grid = np.zeros((4, self.map_size, self.map_size))
        a0, a1 = self.state['agent']
        t0, t1 = self.state['target']
        grid[0, a0, a1] = 1
        grid[1, t0, t1] = 1
        for h in self.state['hazards']:
            grid[2, h[0], h[1]] = 1
        grid[3] = np.array(self.state['rand'])
        return grid

    def _get_info(self):
        """No information necessary for now"""
        return {}

    def reset(self, state=None):
        """Generates a new random map and returns"""
        super().reset()

        if isinstance(state, dict):
            self.state = state
        elif isinstance(state, tuple):
            self.state = {'agent':state[0], 'target':state[1], 'hazards':state[2], 'rand':state[3]}
        elif isinstance(state, np.ndarray):
            agent = tuple(np.argwhere(state[0])[0])
            goal = tuple(np.argwhere(state[1])[0])
            hazards = frozenset(map(tuple, np.argwhere(state[2])))
            rand = tuple(tuple(i) for i in state[3])
            self.state = {'agent':agent, 'target':goal, 'hazards':hazards, 'rand':rand}
        else:
            # Choose the map uniformly at random
            _, self.state = generate_random_map(self.map_size, self.hazard_p, self.seed)

        observation = self.get_state()
        info = self._get_info()

        return observation, info
    
    def next_agents(self, action):
        probs = [1-self.slip_p, self.slip_p/2, self.slip_p/2]
        
        # Sometimes, the agent slips and performs a different action than intended
        # This is controlled by the slip_p parameter
        directions = [
            self._action_to_direction[action],
            self._action_to_direction[(action - 1) % self.action_size],
            self._action_to_direction[(action + 1) % self.action_size],
        ]
        next_agents = [
            tuple(np.clip(np.array(self.state['agent']) + direction, 0, self.map_size - 1)) for direction in directions
        ]
        
        return probs, next_agents

    def step(self, action=None, next_agent=None):
        """Takes a step in the environment, as specified by the action or next_agent"""
        
        if next_agent != None:
            self.state['agent'] = next_agent
        else:
            probs, next_agents = self.next_agents(action)
            # print(next_agents)
            self.state['agent'] = next_agents[np.random.choice(len(next_agents), p=probs)]
            # print(self.state['agent'], type(self.state['agent']))
            # self.state['agent'] = next_agent

        # Checks if the agent has landed on the target or a hazard
        reached_target = self.state['agent'] == self.state['target']
        reached_hazard = self.state['agent'] in self.state['hazards']

        observation = self.get_state()
        # reward = reached_target  # Reward is 1 if reached_target, else 0

        a0, a1 = self.state['agent']
        reward = self.state['rand'][a0][a1]
        if reached_target:
            reward += GOAL
        elif reached_hazard:
            reward += HAZARD
        
        terminated = reached_target or reached_hazard
        truncated = False  # Doesn't truncate
        info = self._get_info()

        return observation, reward, terminated, truncated, info
