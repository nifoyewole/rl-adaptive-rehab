# Baseline policies for comparison against the RL agents.
# Baselines implemented:
#   1. StaticPolicy -- fixed schedule, ignores patient state entirely.
#   2. HeuristicPolicy -- rule-based logic mimicking a simplified clinical protocol.
#   3. RandomPolicy -- uniform random actions (sanity check).
# The comparison is what answers our RQ1.

import numpy as np


class StaticPolicy:
    # Fixed therapy schedule: always applies medium difficulty exercise.
    # This represents the current clinical reality for many rehab centres — a pre-set program that doesn't adapt to patient response.
    # It's the primary baseline because it's the status quo being improved upon.

    def __init__(self):
        self.name = "Static Schedule"

    def select_action(self, obs: np.ndarray) -> int:
        return 1

    def reset(self):
        pass

class HeuristicPolicy:
    # Rule-based clinical heuristic.
    # Mimics simplified physiotherapist decision-making:
    #   - If fatigue is high (>65%), prescribe rest.
    #   - If fatigue is moderate (40-65%) and ROM is low, use low difficulty.
    #   - If fatigue is low (<40%) and ROM is improving, use high difficulty.
    #   - Otherwise, use medium difficulty.
    # This is a stronger baseline than static — it's what a "reasonable"
    # non-adaptive system would do. RL needs to beat this to be interesting.
    # Obs vector: [rom/100, accuracy/100, fatigue/100, stage/3]

    def __init__(self, fatigue_high_threshold=0.65, fatigue_mod_threshold=0.40):
        self.name = "Heuristic Clinical Policy"
        self.fatigue_high = fatigue_high_threshold
        self.fatigue_mod = fatigue_mod_threshold

    def select_action(self, obs: np.ndarray) -> int:
        rom_norm = obs[0]  # 0-1
        fatigue_norm = obs[2]  # 0-1

        if fatigue_norm > self.fatigue_high:
            return 3  # rest -- patient is too fatigued to benefit from exercise

        if fatigue_norm > self.fatigue_mod and rom_norm < 0.5:
            return 0  # low difficulty -- tired + still impaired

        if fatigue_norm < self.fatigue_mod and rom_norm > 0.6:
            return 2  # high difficulty -- fresh + already progressing well

        return 1  # default: medium difficulty

    def reset(self):
        pass


class RandomPolicy:
    # Uniform random action selection.
    # Used as a sanity-check lower bound. If RL can't beat this,
    # something is wrong with the reward function or training.

    def __init__(self, seed=None):
        self.name = "Random Policy"
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, 4))

    def reset(self):
        pass
