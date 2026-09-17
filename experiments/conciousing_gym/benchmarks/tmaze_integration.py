# experiments/conciousing_gym/benchmarks/tmaze_integration.py
"""T-Maze with delayed reward — tests multi-timescale conciousing (K≥3)."""
from __future__ import annotations
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Dict
import chex

@dataclass
class TMazeConfig:
    corridor_length: int = 10
    delay: int = 20          # Steps between cue and choice point
    cue_dim: int = 2         # Left/Right cue at start
    reward_magnitude: float = 1.0

class TMazeEnv:
    """Minimal T-Maze for conciousing evaluation."""
    def __init__(self, cfg: TMazeConfig):
        self.cfg = cfg
        self.state = 0
        self.cue = None
        self.t = 0

    def reset(self, key) -> Tuple[chex.Array, Dict]:
        self.cue = jax.random.randint(key, (), 0, 2)  # 0=left, 1=right
        self.state = 0
        self.t = 0
        obs = self._obs()
        return obs, {"cue": self.cue}

    def step(self, action: int) -> Tuple[chex.Array, float, bool, Dict]:
        self.t += 1
        reward = 0.0
        done = False

        if self.state < self.cfg.corridor_length:
            self.state += 1
        elif self.state == self.cfg.corridor_length:
            # Choice point
            correct = (action == self.cue)
            reward = self.cfg.reward_magnitude if correct else -self.cfg.reward_magnitude
            done = True
        obs = self._obs()
        return obs, reward, done, {}

    def _obs(self) -> chex.Array:
        """Observation: [position_onehot, cue_if_at_start, blank_otherwise]"""
        pos_oh = jnp.zeros(self.cfg.corridor_length + 1)
        pos_oh = pos_oh.at[self.state].set(1.0)
        cue_vec = jnp.zeros(2)
        if self.state == 0:
            cue_vec = cue_vec.at[self.cue].set(1.0)
        return jnp.concatenate([pos_oh, cue_vec])

def conciousing_intensity(profile: list) -> float:
    """∫ ι_t dt over episode — primary metric."""
    return sum(p['integration_proxy'] for p in profile) / len(profile)

def flexibility_index(profile: list) -> float:
    """Variance of sim_horizon — measures adaptive counterfactual depth."""
    horizons = jnp.array([p['sim_horizon'] for p in profile])
    return float(jnp.var(horizons))

def sample_efficiency(profile: list, returns: list) -> float:
    """Area under return curve normalized by conciousing compute."""
    # Proxy: total_imagined_steps per unit return
    total_sim = sum(p['sim_horizon'] for p in profile)
    total_return = sum(returns)
    return total_return / (total_sim + 1e-6)