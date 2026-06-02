import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import uniform_filter1d  # for smooth curves


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Consistent colour palette across all plots
COLOURS = {
    "Static Schedule": "#6B7280",
    "Heuristic Clinical Policy": "#3B82F6",
    "Random Policy": "#EF4444",
    "Q-Learning": "#10B981",
    "DQN": "#8B5CF6",
}


LABELS = {
    "Static Schedule": "Static",
    "Heuristic Clinical Policy": "Heuristic",
    "Random Policy": "Random",
    "Q-Learning": "Q-Learning",
    "DQN": "DQN",
}

THERAPY_COST_PER_SESSION = 120.0


def load_results(profile: str) -> dict:
    path = RESULTS_DIR / f"{profile}_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No results found for profile '{profile}'. "
            f"Run training first: python -m src.training.train --profile {profile}"
        )
    with open(path) as f:
        return json.load(f)


def smooth(values: list[float], window: int = 30) -> np.ndarray:
    """Apply uniform moving average to smooth noisy learning curves."""
    arr = np.array(values, dtype=float)
    return uniform_filter1d(arr, size=window)


# Figure 1: Learning curves
def plot_learning_curves(data: dict, save: bool = True) -> plt.Figure:
    profile = data["profile"]
    results = data["results"]
    rl_agents = ["Q-Learning", "DQN"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Learning curves — {profile.capitalize()} patient profile",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, agent_name in zip(axes, rl_agents):
        if agent_name not in results:
            continue
        r = results[agent_name]
        rewards = r["rewards"]
        episodes = np.arange(1, len(rewards) + 1)

        # Raw (faint)
        ax.plot(episodes, rewards, color=COLOURS[agent_name], alpha=0.15, linewidth=0.7)
        # Smoothed
        ax.plot(
            episodes,
            smooth(rewards),
            color=COLOURS[agent_name],
            linewidth=2,
            label=f"{agent_name} (smoothed)",
        )

        # Overlay baselines as horizontal dashed lines for context
        for bname in ["Static Schedule", "Heuristic Clinical Policy"]:
            if bname in results:
                ax.axhline(
                    results[bname]["mean_reward"],
                    color=COLOURS[bname],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                    label=LABELS[bname],
                )

        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Episode reward", fontsize=11)
        ax.set_title(agent_name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{profile}_learning_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# Figure 2: Policy comparison bar chart


def plot_policy_comparison(data: dict, save: bool = True) -> plt.Figure:
    profile = data["profile"]
    results = data["results"]

    policies = list(results.keys())
    mean_rewards = [results[p]["mean_reward"] for p in policies]
    success_rates = [results[p]["success_rate"] * 100 for p in policies]
    colours = [COLOURS.get(p, "#9CA3AF") for p in policies]
    labels = [LABELS.get(p, p) for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Policy comparison — {profile.capitalize()} patient profile",
        fontsize=13,
        fontweight="bold",
    )

    # Mean reward
    bars = ax1.bar(
        labels, mean_rewards, color=colours, edgecolor="white", linewidth=0.8
    )
    ax1.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax1.set_ylabel("Mean episode reward", fontsize=11)
    ax1.set_title("Mean Cumulative Reward", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    # Success rate
    bars2 = ax2.bar(
        labels, success_rates, color=colours, edgecolor="white", linewidth=0.8
    )
    ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.set_ylabel("Success rate (%)", fontsize=11)
    ax2.set_title("Recovery Threshold Success Rate", fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"{profile}_policy_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# Figure 3: Cross-profile robustness (requires all 3 profiles run)
def plot_cross_profile(save: bool = True) -> plt.Figure | None:
    profiles = ["mild", "moderate", "severe"]
    available = [p for p in profiles if (RESULTS_DIR / f"{p}_results.json").exists()]

    if len(available) < 2:
        print("Need at least 2 profiles to compare. Run --all first.")
        return None

    agents = ["Q-Learning", "DQN"]
    fig, axes = plt.subplots(1, len(agents), figsize=(10, 4))
    if len(agents) == 1:
        axes = [axes]

    fig.suptitle(
        "Cross-profile robustness — mean reward by severity",
        fontsize=13,
        fontweight="bold",
    )

    for ax, agent in zip(axes, agents):
        vals, errs, prof_labels = [], [], []
        for p in available:
            d = load_results(p)
            r = d["results"].get(agent)
            if r:
                vals.append(r["mean_reward"])
                errs.append(r["std_reward"])
                prof_labels.append(p.capitalize())

        ax.bar(
            prof_labels,
            vals,
            yerr=errs,
            color=COLOURS[agent],
            alpha=0.85,
            capsize=4,
            edgecolor="white",
        )
        ax.set_title(agent, fontsize=12)
        ax.set_ylabel("Mean reward ± SD", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "cross_profile_robustness.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


# Business metrics
def compute_business_metrics(data: dict) -> dict:
    """
    Translate simulation results into business-facing metrics.

    Answers RQ3 (cost efficiency) and supports the commercialisation
    section of the report.

    Assumptions:
      - Baseline for comparison = Static Schedule (status quo)
      - Each step in simulation ≈ 5 minutes of therapy time
      - Session cost = €120 (Irish private clinic rate)
      - Recovery threshold hit = successful discharge
    """
    results = data["results"]
    profile = data["profile"]
    baseline = results.get("Static Schedule", {})

    metrics = {}
    baseline_steps = baseline.get("mean_steps", 1)
    baseline_cost = baseline_steps * (THERAPY_COST_PER_SESSION / 50)  # cost per step

    for name, r in results.items():
        steps = r["mean_steps"]
        cost = steps * (THERAPY_COST_PER_SESSION / 50)
        savings = baseline_cost - cost
        efficiency = (
            ((baseline_steps - steps) / baseline_steps * 100) if baseline_steps else 0
        )

        metrics[name] = {
            "mean_steps": round(steps, 1),
            "estimated_cost_eur": round(cost, 2),
            "cost_saving_vs_static_eur": round(savings, 2),
            "efficiency_gain_pct": round(efficiency, 1),
            "success_rate_pct": round(r["success_rate"] * 100, 1),
        }

    return {"profile": profile, "metrics": metrics}


def print_business_table(biz: dict):
    m = biz["metrics"]
    print(f"\nBusiness metrics — {biz['profile'].capitalize()} profile")
    print(f"{'─'*75}")
    print(
        f"{'Policy':<30} {'Steps':>7} {'Cost (€)':>10} {'Saving (€)':>11} {'Efficiency':>11} {'Success':>8}"
    )
    print(f"{'─'*75}")
    for name, v in m.items():
        print(
            f"{LABELS.get(name, name):<30} "
            f"{v['mean_steps']:>7.1f} "
            f"{v['estimated_cost_eur']:>10.2f} "
            f"{v['cost_saving_vs_static_eur']:>11.2f} "
            f"{v['efficiency_gain_pct']:>10.1f}% "
            f"{v['success_rate_pct']:>7.1f}%"
        )
    print(f"{'─'*75}")


# Figure 4: Ablation study visualisation
def plot_ablation(save: bool = True) -> plt.Figure | None:
    """
    Visualise the reward function ablation study results.
    Generates two charts side by side:
      Left:  Success rate per configuration (primary clinical metric)
      Right: Final fatigue per configuration (safety metric)
    The juxtaposition makes the safety-efficacy tradeoff immediately visible.
    """
    path = RESULTS_DIR / "ablation_results.json"
    if not path.exists():
        print("No ablation results found. Run: python -m src.training.ablation")
        return None

    with open(path) as f:
        data = json.load(f)

    configs = data["ablation"]
    names = [c["name"] for c in configs]
    success = [c["success_rate"] for c in configs]
    fatigue = [c["mean_final_fatigue"] for c in configs]
    costs = [c["est_cost_eur"] for c in configs]
    stability = [c["late_reward_std"] for c in configs]

    # Short labels for chart x-axis
    short_labels = [
        "Full model\n(proposed)",
        "No fatigue\n(\u03b2=0)",
        "No cost\n(\u03b3=0)",
        "High cost\n(\u03b3=0.3)",
        "Recovery\nonly",
    ]

    # Colour: highlight full model, mute others
    bar_colours = ["#10B981" if i == 0 else "#D1D5DB" for i in range(len(configs))]
    fat_colours = ["#EF4444" if i == 0 else "#D1D5DB" for i in range(len(configs))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Reward Function Ablation Study — DQN on Moderate Profile (1,000 Episodes)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # --- Left: Success rate ---
    ax = axes[0]
    bars = ax.bar(
        short_labels, success, color=bar_colours, edgecolor="white", linewidth=0.8
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Recovery success rate (%)", fontsize=11)
    ax.set_title("Recovery Success Rate", fontsize=12)
    ax.set_ylim(0, max(success) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)

    # --- Middle: Final fatigue ---
    ax = axes[1]
    bars2 = ax.bar(
        short_labels, fatigue, color=fat_colours, edgecolor="white", linewidth=0.8
    )
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    # Safe fatigue threshold line
    ax.axhline(
        y=80,
        color="#EF4444",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Overtraining risk threshold",
    )
    ax.set_ylabel("Mean final fatigue (0–100)", fontsize=11)
    ax.set_title("Patient Fatigue Level", fontsize=12)
    ax.set_ylim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=8)

    # --- Right: Policy stability (σ of late-episode rewards) ---
    ax = axes[2]
    stab_colours = ["#8B5CF6" if i == 0 else "#D1D5DB" for i in range(len(configs))]
    bars3 = ax.bar(
        short_labels, stability, color=stab_colours, edgecolor="white", linewidth=0.8
    )
    ax.bar_label(bars3, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Policy variance \u03c3 (late episodes)", fontsize=11)
    ax.set_title("Policy Stability", fontsize=12)
    ax.set_ylim(0, max(stability) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()

    if save:
        out = RESULTS_DIR / "ablation_results.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    return fig


# Entry point
def generate_all_plots(profile: str):
    data = load_results(profile)
    plot_learning_curves(data)
    plot_policy_comparison(data)
    biz = compute_business_metrics(data)
    print_business_table(biz)
    # Save business metrics
    path = RESULTS_DIR / f"{profile}_business_metrics.json"
    with open(path, "w") as f:
        json.dump(biz, f, indent=2)
    print(f"Business metrics saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="moderate",
        choices=["mild", "moderate", "severe"],
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Generate ablation study chart from results/ablation_results.json",
    )
    args = parser.parse_args()

    if args.ablation:
        plot_ablation()
    elif args.all:
        for p in ["mild", "moderate", "severe"]:
            try:
                generate_all_plots(p)
            except FileNotFoundError as e:
                print(f"Skipping {p}: {e}")
        plot_cross_profile()
        plot_ablation()
    else:
        generate_all_plots(args.profile)
        plot_cross_profile()
