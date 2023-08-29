from collections import defaultdict
import numpy as np
from tqdm import tqdm


class Qlearning:
    def __init__(self, action_size, learning_rate, gamma):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""

        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state])
            - self.qtable[state][action]
        )

        self.qtable[state][action] += self.learning_rate * delta

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = defaultdict(lambda: np.zeros(self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        explor_exploit_tradeoff = np.random.random()

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # If all Q-values are equal, sample randomly
            if np.all(qtable[state]) == qtable[state][0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state])
        return action


class Experiment:
    def __init__(self, env, learner, explorer, episodes=1000):
        self.env = env
        self.learner = learner
        self.explorer = explorer
        self.episodes = episodes

    def run(self):
        states, steps, rewards, actions = [], [], [], []

        self.learner.reset_qtable()
        for _ in tqdm(range(self.episodes), leave=False):
            state = self.env.reset()[0]
            step_count = 0
            reward_sum = 0
            while True:
                states.append(state)
                action = self.explorer.choose_action(
                    action_space=self.env.action_space,
                    state=state,
                    qtable=self.learner.qtable,
                )
                actions.append(action)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                self.learner.update(state, action, reward, new_state)
                state = new_state
                reward_sum += reward
                step_count += 1

                if terminated or truncated:
                    break
            # states.append(state)
            rewards.append(reward_sum)
            steps.append(step_count)

        return states, steps, rewards, actions, self.learner.qtable
