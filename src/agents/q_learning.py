import numpy as np
import pickle
from pathlib import Path


class QLearningAgent:
    def __init__(
        self,
        n_actions=4,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        n_bins=10,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        self.name = "Q-Learning"

        # Q-table: shape (n_bins, n_bins, n_bins, 4_stages, n_actions)
        # Initialise to small positive values to encourage exploration early
        self.q_table = np.zeros((n_bins, n_bins, n_bins, 4, n_actions))

        # Training bookkeeping
        self.episode_rewards: list[float] = []
        self.epsilons: list[float] = []
        self.training_steps: int = 0

    # Discretisation
    def _discretise(self, obs: np.ndarray) -> tuple:
        """
        Convert continuous observation [rom, acc, fatigue, stage] into
        discrete bin indices for Q-table lookup.
        """
        rom_bin = min(int(obs[0] * self.n_bins), self.n_bins - 1)
        acc_bin = min(int(obs[1] * self.n_bins), self.n_bins - 1)
        fat_bin = min(int(obs[2] * self.n_bins), self.n_bins - 1)
        stage = min(int(obs[3] * 3 + 0.5), 3)  # round to nearest stage 0-3
        return (rom_bin, acc_bin, fat_bin, stage)

    # Action selection

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        During evaluation (training=False), always exploit (greedy).
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        state_idx = self._discretise(obs)
        return int(np.argmax(self.q_table[state_idx]))

    # Learning

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Single Q-learning update step (Bellman equation)."""
        s = self._discretise(obs)
        s_ = self._discretise(next_obs)

        # Current Q estimate
        q_current = self.q_table[s][action]

        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[s_])

        # Update
        self.q_table[s][action] += self.alpha * (td_target - q_current)
        self.training_steps += 1

    def end_episode(self, episode_reward: float):
        """Call at end of each episode to log metrics and decay epsilon."""
        self.episode_rewards.append(episode_reward)
        self.epsilons.append(self.epsilon)
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Reset RNG for evaluation runs — does NOT reset Q-table."""
        pass

    # Persistence
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                    "episode_rewards": self.episode_rewards,
                    "training_steps": self.training_steps,
                },
                f,
            )
        print(f"Q-table saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.episode_rewards = data["episode_rewards"]
        self.training_steps = data["training_steps"]
        print(f"Q-table loaded from {path}")

    # Initialise RNG (called after construction so seed can be set)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (
            f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.3f}, steps={self.training_steps})"
        )
