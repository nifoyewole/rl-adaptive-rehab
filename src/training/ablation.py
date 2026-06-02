import json
import numpy as np
from pathlib import Path
from src.env.patient_env import PatientEnv
import src.env.patient_env as env_module
from src.agents.dqn import DQNAgent

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_EPISODES = 1000
SEED = 42
PROFILE = "moderate"  # ablation always on moderate for controlled comparison


def run_ablation_config(name: str, alpha: float, beta: float, gamma: float) -> dict:
    """Run DQN with specific reward weights and return results."""
    print(f"\n  Running: {name}  (α={alpha}, β={beta}, γ={gamma})")

    env_module.ALPHA = alpha
    env_module.BETA = beta
    env_module.GAMMA = gamma

    env = PatientEnv(profile=PROFILE, max_steps=50, seed=SEED)
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

    import random, torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    steps_list, final_roms, final_fatigue = [], [], []
    success_count = 0

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        ep_reward, steps = 0.0, 0
        ep_losses = []

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.push(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss:
                ep_losses.append(loss)
            obs = next_obs
            ep_reward += reward
            steps += 1
            if done:
                break

        agent.end_episode(ep_reward, np.mean(ep_losses) if ep_losses else None)
        steps_list.append(steps)
        final_roms.append(info["rom"])
        final_fatigue.append(info["fatigue"])
        if info["reached_threshold"]:
            success_count += 1

    # Reward variance in last 200 episodes = stability proxy for RQ2
    last_200 = agent.episode_rewards[-200:]

    return {
        "name": name,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "mean_reward": round(float(np.mean(agent.episode_rewards)), 2),
        "std_reward": round(float(np.std(agent.episode_rewards)), 2),
        "late_reward_std": round(float(np.std(last_200)), 2),  # RQ2: stability
        "success_rate": round(success_count / N_EPISODES * 100, 1),
        "mean_steps": round(float(np.mean(steps_list)), 1),
        "mean_final_rom": round(float(np.mean(final_roms)), 2),
        "mean_final_fatigue": round(float(np.mean(final_fatigue)), 2),
        "est_cost_eur": round(float(np.mean(steps_list)) * (120 / 50), 2),
        "rewards": agent.episode_rewards,
    }


def run_ablation():
    print(f"\n{'='*60}")
    print("Ablation study — DQN reward weight comparison")
    print(f"Profile: {PROFILE} | Episodes: {N_EPISODES}")
    print(f"{'='*60}")

    configs = [
        ("Full model (current)", 2.0, 0.3, 0.1),
        ("No fatigue penalty (β=0)", 2.0, 0.0, 0.1),
        ("No cost penalty (γ=0)", 2.0, 0.3, 0.0),
        ("High cost penalty (γ=0.3)", 2.0, 0.3, 0.3),
        ("Recovery only (β=0, γ=0)", 2.0, 0.0, 0.0),
    ]

    results = []
    for name, alpha, beta, gamma in configs:
        r = run_ablation_config(name, alpha, beta, gamma)
        results.append(r)
        print(
            f"    ✓ {name}: reward={r['mean_reward']:6.1f} | "
            f"success={r['success_rate']}% | "
            f"fatigue={r['mean_final_fatigue']:.1f} | "
            f"cost=€{r['est_cost_eur']:.2f} | "
            f"stability(σ)={r['late_reward_std']:.1f}"
        )

    # Reset to current best weights after ablation
    env_module.ALPHA = 2.0
    env_module.BETA = 0.3
    env_module.GAMMA = 0.1

    # Save (strip raw rewards list to keep file small)
    out = []
    for r in results:
        r_save = {k: v for k, v in r.items() if k != "rewards"}
        out.append(r_save)

    path = RESULTS_DIR / "ablation_results.json"
    with open(path, "w") as f:
        json.dump(
            {"profile": PROFILE, "n_episodes": N_EPISODES, "ablation": out}, f, indent=2
        )
    print(f"\nAblation results saved to {path}")

    # Print summary table
    print(f"\n{'─'*85}")
    print(
        f"{'Config':<30} {'Reward':>8} {'Success':>8} {'Fatigue':>8} "
        f"{'Cost(€)':>9} {'Stability(σ)':>13}"
    )
    print(f"{'─'*85}")
    for r in out:
        print(
            f"{r['name']:<30} {r['mean_reward']:>8.1f} "
            f"{r['success_rate']:>7.1f}% "
            f"{r['mean_final_fatigue']:>8.1f} "
            f"{r['est_cost_eur']:>9.2f} "
            f"{r['late_reward_std']:>13.1f}"
        )
    print(f"{'─'*85}")

    print("\nKey findings:")
    full = next(r for r in out if "Full model" in r["name"])
    no_fat = next(r for r in out if "β=0" in r["name"] and "γ=0" not in r["name"])
    no_cst = next(r for r in out if "γ=0)" in r["name"])
    hi_cst = next(r for r in out if "γ=0.3" in r["name"])

    fat_stability = full["late_reward_std"] - no_fat["late_reward_std"]
    cost_saving = hi_cst["est_cost_eur"] - full["est_cost_eur"]

    print(
        f"  RQ2: Fatigue penalty {'increases' if fat_stability > 0 else 'decreases'} "
        f"policy variance by {abs(fat_stability):.1f} σ units → "
        f"{'more stable' if fat_stability < 0 else 'less stable'} with fatigue modelling"
    )
    print(
        f"  RQ3: High cost penalty (γ=0.3) costs €{abs(cost_saving):.2f} more per episode "
        f"vs current — cost shaping {'reduces' if cost_saving < 0 else 'adds'} resource use"
    )


if __name__ == "__main__":
    run_ablation()
