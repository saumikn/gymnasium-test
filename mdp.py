import numpy as np
import gymnasium as gym
from gymnasium import spaces


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

    valid = False
    while not valid:
        grid = np.zeros((3, size, size), dtype=int)
        p = min(p, 1)

        agent, target = rng.choice(grid[0].size, size=2, replace=False)
        agent = (agent // size, agent % size)
        target = (target // size, target % size)
        grid[0, *agent] = 1
        grid[1, *target] = 1
        grid[2] = rng.choice([0, 1], size=(size, size), p=[1 - p, p])
        grid[2, *agent] = 0  # Agent isn't on a hazard
        grid[2, *target] = 0  # Target isn't on a hazard
        valid = is_valid(grid[2], agent, target)
        hazards = frozenset(map(tuple, np.argwhere(grid[2])))

    return grid, agent, target, hazards


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
        return (self.agent, self.target, self.hazards)

    def set_state(self, agent, target, hazards):
        self.agent = agent
        self.target = target
        self.hazards = hazards

    def get_grid(self):
        grid = np.zeros((3, self.map_size, self.map_size), dtype=bool)
        grid[0, *self.agent] = 1
        grid[1, *self.target] = 1
        for h in self.hazards:
            grid[2, *h] = 1
        return grid

    def _get_info(self):
        """No information necessary for now"""
        return {}

    def reset(self, state=None):
        """Generates a new random map and returns"""
        super().reset()

        if state:
            self.agent, self.target, self.hazards = state
        else:
            # Choose the map uniformly at random
            _, self.agent, self.target, self.hazards = generate_random_map(
                self.map_size, self.hazard_p, self.seed
            )

        observation = self.get_state()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Takes a step in the environment, as specified by the action"""

        # Sometimes, the agent slips and performs a different action than intended
        # This is controlled by the slip_p parameter
        slip = np.random.random()
        if slip > self.slip_p:  # Success
            direction = self._action_to_direction[action]
        elif slip < self.slip_p / 2:  # Slip counterclockwise
            direction = self._action_to_direction[action - 1]
        else:  # Slip clockwise
            direction = self._action_to_direction[(action + 1) % self.action_size]

        # Updates the location of the agent in the grid
        self.agent = tuple(
            np.clip(np.array(self.agent) + direction, 0, self.map_size - 1)
        )

        # Checks if the agent has landed on the target or a hazard
        reached_target = self.agent == self.target
        reached_hazard = self.agent in self.hazards

        observation = self.get_state()
        # reward = reached_target  # Reward is 1 if reached_target, else 0
        reward = 10 if reached_target else -1 if reached_hazard else 0
        terminated = reached_target or reached_hazard
        truncated = False  # Doesn't truncate
        info = self._get_info()

        return observation, reward, terminated, truncated, info
