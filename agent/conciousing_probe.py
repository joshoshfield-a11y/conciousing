# conciousing_probe.py
# Instrument any agent loop to log C_t = (b_t, π_t, G_t, P_t, i_t)

import torch, time, json, numpy as np
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class ConciousingState:
    t: int
    wall_time: float
    belief_entropy: float          # H[b_t] — uncertainty
    policy_entropy: float          # H[π_t] — exploration
    goal_stack_depth: int          # len(𝒢_t)
    sim_horizon: int               # max depth of 𝒫_t rollouts
    integration_proxy: float       # e.g. MI(b_t; π_t) approx or Φ proxy
    surprise: float                # -log p(o_t | b_{t-1})

class ConciousingProbe:
    def __init__(self, agent, belief_extractor, policy_extractor,
                 goal_stack_getter, sim_horizon_getter, integration_estimator):
        self.agent = agent
        self.belief_extractor = belief_extractor
        self.policy_extractor = policy_extractor
        self.goal_stack_getter = goal_stack_getter
        self.sim_horizon_getter = sim_horizon_getter
        self.integration_estimator = integration_estimator
        self.log: List[ConciousingState] = []

    def step(self, obs, action, reward):
        b = self.belief_extractor(self.agent)
        π = self.policy_extractor(self.agent)
        G = self.goal_stack_getter(self.agent)
        H = self.sim_horizon_getter(self.agent)
        ι = self.integration_estimator(b, π)
        surp = -self.agent.world_model.log_prob(obs, b.prev).item() if hasattr(self.agent, 'world_model') else 0.0

        cs = ConciousingState(
            t=len(self.log),
            wall_time=time.time(),
            belief_entropy=-(b.log_prob * b.prob).sum().item() if hasattr(b, 'log_prob') else 0.0,
            policy_entropy=-(π.logits * π.probs).sum(-1).mean().item() if hasattr(π, 'logits') else 0.0,
            goal_stack_depth=len(G),
            sim_horizon=H,
            integration_proxy=ι,
            surprise=surp
        )
        self.log.append(cs)
        return cs

    def save_jsonl(self, path):
        with open(path, 'w') as f:
            for cs in self.log:
                f.write(json.dumps(asdict(cs)) + '\n')