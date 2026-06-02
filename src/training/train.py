import numpy as np
import json
import time
from pathlib import Path

from src.env.patient_env import PatientEnv
from src.env.patient_profiles import PROFILES
from src.agents.baselines import StaticPolicy, HeuristicPolicy, RandomPolicy
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# Core training functions
def run_baseline(policy, env: PatientEnv, n_episodes: int, seed: int = 42) -> dict:
    """
    Evaluate a baseline policy for n_episodes.
    Baselines don't learn — we're just recording their performance.
    """
    rewards = []
    steps_to_done = []
    success_count = 0
    final_roms = []
    final_fatigue = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        policy.reset()
        ep_reward = 0.0
        steps = 0

        while True:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        steps_to_done.append(steps)
        final_roms.append(info["rom"])
        final_fatigue.append(info["fatigue"])
        if info["reached_threshold"]:
            success_count += 1

    return {
        "policy": policy.name,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_to_done)),
        "success_rate": success_count / n_episodes,
        "mean_final_rom": float(np.mean(final_roms)),
        "mean_final_fatigue": float(np.mean(final_fatigue)),
    }


def train_q_learning(
    env: PatientEnv,
    n_episodes: int = 500,
    seed: int = 42,
    verbose_every: int = 100,
) -> tuple[QLearningAgent, dict]:
    """Train tabular Q-learning agent."""

    agent = QLearningAgent(
        n_actions=4,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        n_bins=10,
    )
    agent.set_seed(seed)

    rng = np.random.default_rng(seed)
    steps_to_done = []
    success_count = 0
    final_roms = []
    final_fatigue = []

    start = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        ep_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                break

        agent.end_episode(ep_reward)
        steps_to_done.append(steps)
        final_roms.append(info["rom"])
        final_fatigue.append(info["fatigue"])
        if info["reached_threshold"]:
            success_count += 1

        if verbose_every and (ep + 1) % verbose_every == 0:
            recent = agent.episode_rewards[-verbose_every:]
            print(
                f"  [Q-Learn] Ep {ep+1:4d}/{n_episodes} | "
                f"Avg reward (last {verbose_every}): {np.mean(recent):6.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start
    print(f"  Q-Learning training done in {elapsed:.1f}s")

    results = {
        "policy": agent.name,
        "rewards": agent.episode_rewards,
        "mean_reward": float(np.mean(agent.episode_rewards)),
        "std_reward": float(np.std(agent.episode_rewards)),
        "mean_steps": float(np.mean(steps_to_done)),
        "success_rate": success_count / n_episodes,
        "mean_final_rom": float(np.mean(final_roms)),
        "mean_final_fatigue": float(np.mean(final_fatigue)),
        "epsilons": agent.epsilons,
    }
    return agent, results


def train_dqn(
    env: PatientEnv,
    n_episodes: int = 500,
    seed: int = 42,
    verbose_every: int = 100,
) -> tuple[DQNAgent, dict]:
    """Train DQN agent."""

    agent = DQNAgent(
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
    )

    random_seed = seed
    import random as stdlib_random
    import torch

    stdlib_random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rng = np.random.default_rng(seed)
    steps_to_done = []
    success_count = 0
    final_roms = []
    final_fatigue = []
    episode_losses = []

    start = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        ep_reward = 0.0
        ep_losses = []
        steps = 0

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.push(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            ep_reward += reward
            steps += 1

            if done:
                break

        mean_loss = float(np.mean(ep_losses)) if ep_losses else None
        agent.end_episode(ep_reward, mean_loss)
        steps_to_done.append(steps)
        final_roms.append(info["rom"])
        final_fatigue.append(info["fatigue"])
        episode_losses.append(mean_loss or 0.0)
        if info["reached_threshold"]:
            success_count += 1

        if verbose_every and (ep + 1) % verbose_every == 0:
            recent = agent.episode_rewards[-verbose_every:]
            print(
                f"  [DQN]     Ep {ep+1:4d}/{n_episodes} | "
                f"Avg reward (last {verbose_every}): {np.mean(recent):6.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start
    print(f"  DQN training done in {elapsed:.1f}s")

    results = {
        "policy": agent.name,
        "rewards": agent.episode_rewards,
        "mean_reward": float(np.mean(agent.episode_rewards)),
        "std_reward": float(np.std(agent.episode_rewards)),
        "mean_steps": float(np.mean(steps_to_done)),
        "success_rate": success_count / n_episodes,
        "mean_final_rom": float(np.mean(final_roms)),
        "mean_final_fatigue": float(np.mean(final_fatigue)),
        "epsilons": agent.epsilons,
        "losses": episode_losses,
    }
    return agent, results


# Full experiment runner
def run_full_experiment(
    profile: str = "moderate",
    n_episodes: int = 500,
    seed: int = 42,
    save_results: bool = True,
) -> dict:
    """
    Run all policies on a given patient profile and return all results.

    This is the main entry point called by the evaluation and plotting modules.
    Results are saved to results/<profile>_results.json.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: profile={profile}, episodes={n_episodes}")
    print(f"{'='*60}")

    env = PatientEnv(profile=profile, max_steps=50, seed=seed)
    all_results = {}

    # --- Baselines ---
    print("\nEvaluating baselines...")
    for policy in [StaticPolicy(), HeuristicPolicy(), RandomPolicy(seed=seed)]:
        print(f"  Running {policy.name}...")
        all_results[policy.name] = run_baseline(policy, env, n_episodes, seed)

    # --- Q-Learning ---
    print("\nTraining Q-Learning...")
    q_agent, q_results = train_q_learning(env, n_episodes, seed)
    all_results["Q-Learning"] = q_results

    # --- DQN ---
    print("\nTraining DQN...")
    dqn_agent, dqn_results = train_dqn(env, n_episodes, seed)
    all_results["DQN"] = dqn_results

    # Summary table
    print(f"\n{'─'*60}")
    print(f"{'Policy':<30} {'Mean Reward':>12} {'Success Rate':>13} {'Final ROM':>10}")
    print(f"{'─'*60}")
    for name, r in all_results.items():
        print(
            f"{name:<30} {r['mean_reward']:>12.2f} "
            f"{r['success_rate']:>12.1%} "
            f"{r['mean_final_rom']:>9.1f}"
        )
    print(f"{'─'*60}")

    if save_results:
        out = {
            "profile": profile,
            "n_episodes": n_episodes,
            "seed": seed,
            "results": all_results,
        }
        path = RESULTS_DIR / f"{profile}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {path}")

        # Save trained models
        q_agent.save(str(RESULTS_DIR / f"{profile}_q_table.pkl"))
        dqn_agent.save(str(RESULTS_DIR / f"{profile}_dqn.pt"))

    return all_results


# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate rehabilitation RL agents"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="moderate",
        choices=["mild", "moderate", "severe"],
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all three profiles")
    args = parser.parse_args()

    if args.all:
        for p in ["mild", "moderate", "severe"]:
            run_full_experiment(p, args.episodes, args.seed)
    else:
        run_full_experiment(args.profile, args.episodes, args.seed)
