import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# Neural network
class QNetwork(nn.Module):
    """
    Two-layer MLP Q-function approximator.
    Input: state vector (obs_dim,)
    Output: Q-values for each action (n_actions,)
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Replay buffer
class ReplayBuffer:
    """
    Fixed-size circular buffer storing experience tuples.
    Random sampling decorrelates training data.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    # DQN agent with experience replay and target network.
    def __init__(
        self,
        obs_dim=4,
        n_actions=4,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10_000,
        batch_size=64,
        target_update_freq=10,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.name = "DQN"

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # Optimiser
        self.optimiser = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Bookkeeping
        self.episode_rewards: list[float] = []
        self.losses: list[float] = []
        self.epsilons: list[float] = []
        self.episode_count: int = 0
        self.training_steps: int = 0

    # Action selection

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # Learning

    def push(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient update.
        Returns loss value (for logging), or None if buffer not ready.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values for taken actions
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (using frozen target network)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimiser.step()

        self.training_steps += 1
        return float(loss.item())

    def end_episode(self, episode_reward: float, loss: float | None = None):
        """Call at end of each episode."""
        self.episode_rewards.append(episode_reward)
        self.epsilons.append(self.epsilon)
        if loss is not None:
            self.losses.append(loss)

        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def reset(self):
        pass

    # Persistence
    def save(self, path: str):
        torch.save(
            {
                "q_net_state": self.q_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "epsilon": self.epsilon,
                "episode_rewards": self.episode_rewards,
                "episode_count": self.episode_count,
            },
            path,
        )
        print(f"DQN model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_count = checkpoint["episode_count"]
        print(f"DQN model loaded from {path}")

    def __repr__(self):
        return (
            f"DQNAgent(γ={self.gamma}, ε={self.epsilon:.3f}, "
            f"buffer={len(self.buffer)}, steps={self.training_steps})"
        )
