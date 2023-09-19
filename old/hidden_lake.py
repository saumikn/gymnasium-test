import numpy as np
import gymnasium as gym
from gymnasium import spaces


def add_visible(grid, square):
    """Marks the square and the four points adjacant as visible."""
    s0, s1 = square
    size = grid.shape[0]
    for r, c in [
        (s0, s1),
        (s0 + 1, s1),
        (s0 - 1, s1),
        (s0, s1 + 1),
        (s0, s1 - 1),
    ]:
        if 0 <= r < size and 0 <= c < size:
            grid[r, c] = 1


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


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    while not valid:
        grid = np.zeros((4, size, size))
        p = min(p, 1)
        agent = (0, 0)
        target = np.random.choice(size * size - 1) + 1
        target = (target // size, target % size)
        grid[0, *agent] = 1
        grid[1, *target] = 1
        grid[2] = np.random.choice([0, 1], size=(size, size), p=[1 - p, p])
        grid[2, *agent] = 0  # Agent isn't on a hazard
        grid[2, *target] = 0  # Target isn't on a hazard
        valid = is_valid(grid[2], agent, target)

    add_visible(grid[3], agent)

    return grid


class HiddenLakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_size=4, hazard_p=0.8, slip_p=0):
        self.render_mode = None
        self.map_size = map_size
        self.hazard_p = hazard_p
        self.slip_p = slip_p
        self.action_size = 4  # Hardcoded to Up, Down, Left, Right

        """
        The obversation space is a n by n grid.
        The first channel represents the location of the agent.
        The second channel represents the location of the target.
        The third channel represents the location of the hazards.
        The fourth channel represents the visible squares in the map.
        """
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.map_size, self.map_size), dtype=bool
        )

        self.action_space = spaces.Discrete(self.action_size)
        self._action_to_direction = [
            np.array([1, 0]),  # Down
            np.array([0, 1]),  # Right
            np.array([-1, 0]),  # Up
            np.array([0, -1]),  # Left
        ]

    def _get_obs(self):
        def totuple(a):
            """Turns 3d Numpy Array into tuple of tuples of tuples, for hashability"""
            return tuple(tuple(tuple(row) for row in matrix) for matrix in a)

        return totuple(self.grid[:3] * np.array([self.grid[3]]))

    def _get_info(self):
        """No information necessary for now"""
        return {}

    def _add_visible(self, square):
        """
        Marks the four points adjacant to square and the square itself as visible
        on the grid.
        """
        s0, s1 = square
        for r, c in [
            (s0, s1),
            (s0 + 1, s1),
            (s0 - 1, s1),
            (s0, s1 + 1),
            (s0, s1 - 1),
        ]:
            if 0 <= r < self.map_size and 0 <= c < self.map_size:
                self.grid[3, r, c] = 1

    def reset(self, options=None):
        """Generates a new random map and returns"""
        super().reset()

        # Choose the map uniformly at random
        self.grid = generate_random_map(self.map_size, self.hazard_p)
        observation = self._get_obs()
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
        cur_agent = np.argwhere(self.grid[0])[0]
        new_agent = cur_agent + direction
        new_agent = np.clip(new_agent, 0, self.map_size - 1)
        self.grid[0] = 0
        self.grid[0, *new_agent] = 1

        # Checks if the agent has landed on the target or a hazard
        reached_target = np.sum(self.grid[0] * self.grid[1])
        reached_hazard = np.sum(self.grid[0] * self.grid[2])

        # Increases the visibility of the grid after moving.
        add_visible(self.grid[3], new_agent)

        observation = self._get_obs()
        reward = reached_target  # Reward is 1 if reached_target, else 0
        terminated = reached_target or reached_hazard
        truncated = False  # Doesn't truncate
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def get_state(self):
        return self.grid

    def set_state(self, grid):
        self.grid = grid
