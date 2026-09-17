# hai_conciousing_agent.py
# JAX implementation (primary) + PyTorch fallback.
# Run: python hai_conciousing_agent.py --backend jax --steps 5000 --profile-out profile.jsonl
# Dependencies: jax, jaxlib, optax, distrax, flax, numpy, matplotlib (for live plot)

from __future__ import annotations
import argparse, json, time, sys, os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np

# ===== Backend Selection =====
BACKEND = None
try:
    import jax, jax.numpy as jnp
    from jax import random, grad, jit, vmap, value_and_grad, lax
    import optax
    import distrax
    BACKEND = "jax"
except ImportError:
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
        import torch.distributions as D
        BACKEND = "torch"
    except ImportError:
        raise RuntimeError("Install jax/jaxlib or torch")

# ===== Configuration =====
@dataclass
class HAIConfig:
    # Environment
    obs_dim: int = 4
    action_dim: int = 2
    # Latent hierarchy
    latent_dims: Tuple[int, ...] = (16, 8, 4)          # K=3 levels
    timescales: Tuple[int, ...] = (1, 5, 20)           # τ₁, τ₂, τ₃
    # Dynamics
    transition_hidden: int = 64
    obs_hidden: int = 64
    # Policy
    policy_hidden: int = 64
    # Planning
    plan_horizon: int = 10
    n_particles: int = 16
    # Learning
    lr: float = 3e-4
    kl_weight: float = 1.0
    # Profiling
    profile_every: int = 1
    estimator: str = "FE"  # "FE" | "MI" | "Phi"

# ===== Core Distributions (Backend-Agnostic API) =====
class DistAPI:
    @staticmethod
    def normal(mean, std): ...
    @staticmethod
    def kl_divergence(p, q): ...
    @staticmethod
    def entropy(p): ...
    @staticmethod
    def sample(p, key, n): ...

if BACKEND == "jax":
    class JaxDistAPI(DistAPI):
        @staticmethod
        def normal(mean, std): return distrax.Normal(mean, std)
        @staticmethod
        def kl_divergence(p, q): return distrax.kl_divergence(p, q)
        @staticmethod
        def entropy(p): return p.entropy()
        @staticmethod
        def sample(p, key, n): return p.sample(seed=key, sample_shape=(n,))
    Dist = JaxDistAPI()
elif BACKEND == "torch":
    class TorchDistAPI(DistAPI):
        @staticmethod
        def normal(mean, std): return D.Normal(mean, std)
        @staticmethod
        def kl_divergence(p, q): return D.kl_divergence(p, q)
        @staticmethod
        def entropy(p): return p.entropy()
        @staticmethod
        def sample(p, key, n): return p.sample((n,))
    Dist = TorchDistAPI()

# ===== Neural Modules =====
if BACKEND == "jax":
    from flax import linen as nn
    class MLP(nn.Module):
        features: Tuple[int, ...]
        activate_final: bool = False
        @nn.compact
        def __call__(self, x):
            for i, feat in enumerate(self.features):
                x = nn.Dense(feat, name=f"dense_{i}")(x)
                if i < len(self.features)-1 or self.activate_final:
                    x = nn.swish(x)
            return x
else:
    class MLP(nn.Module):
        def __init__(self, features: List[int], activate_final=False):
            super().__init__()
            layers = []
            for i, (in_f, out_f) in enumerate(zip(features[:-1], features[1:])):
                layers.append(nn.Linear(in_f, out_f))
                if i < len(features)-2 or activate_final:
                    layers.append(nn.SiLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

# ===== World Model per Level =====
if BACKEND == "jax":
    class LevelWorldModel(nn.Module):
        latent_dim: int
        obs_dim: int
        action_dim: int
        hidden: int
        @nn.compact
        def __call__(self, z, a, key):
            # Transition: p(z' | z, a)
            za = jnp.concatenate([z, a], -1)
            h = MLP([self.hidden, self.hidden, self.latent_dim*2], name="trans")(za)
            mu_z, logvar_z = jnp.split(h, 2, -1)
            trans_dist = Dist.normal(mu_z, jnp.exp(0.5*logvar_z) + 1e-4)
            # Observation: p(o | z)
            h_o = MLP([self.hidden, self.hidden, self.obs_dim*2], name="obs")(z)
            mu_o, logvar_o = jnp.split(h_o, 2, -1)
            obs_dist = Dist.normal(mu_o, jnp.exp(0.5*logvar_o) + 1e-4)
            return trans_dist, obs_dist
else:
    class LevelWorldModel(nn.Module):
        def __init__(self, latent_dim, obs_dim, action_dim, hidden):
            super().__init__()
            self.latent_dim = latent_dim
            self.trans_net = MLP([latent_dim+action_dim, hidden, hidden, latent_dim*2])
            self.obs_net = MLP([latent_dim, hidden, hidden, obs_dim*2])
        def forward(self, z, a):
            za = torch.cat([z, a], -1)
            h = self.trans_net(za)
            mu_z, logvar_z = h.chunk(2, -1)
            trans_dist = Dist.normal(mu_z, torch.exp(0.5*logvar_z)+1e-4)
            h_o = self.obs_net(z)
            mu_o, logvar_o = h_o.chunk(2, -1)
            obs_dist = Dist.normal(mu_o, torch.exp(0.5*logvar_o)+1e-4)
            return trans_dist, obs_dist

# ===== Hierarchical Active Inference Agent =====
if BACKEND == "jax":
    class HAIAgent(nn.Module):
        config: HAIConfig
        def setup(self):
            self.levels = [LevelWorldModel(
                latent_dim=d, obs_dim=self.config.obs_dim if k==0 else self.config.latent_dims[k-1],
                action_dim=self.config.action_dim if k==0 else self.config.latent_dims[k-1],
                hidden=self.config.transition_hidden
            ) for k, d in enumerate(self.config.latent_dims)]
            self.policy_nets = [MLP([d, self.config.policy_hidden, self.config.policy_hidden, self.config.action_dim*2])
                                for d in self.config.latent_dims]
            # Variational params (mean-field Gaussian)
            self.belief_means = [self.param(f"bel_mu_{k}", nn.initializers.zeros, (d,)) for k,d in enumerate(self.config.latent_dims)]
            self.belief_logvars = [self.param(f"bel_logvar_{k}", nn.initializers.zeros, (d,)) for k,d in enumerate(self.config.latent_dims)]
            # Goal priors (learnable target distributions)
            self.goal_means = [self.param(f"goal_mu_{k}", nn.initializers.zeros, (d,)) for k,d in enumerate(self.config.latent_dims)]
            self.goal_logvars = [self.param(f"goal_logvar_{k}", nn.initializers.zeros, (d,)) for k,d in enumerate(self.config.latent_dims)]
            self.optim = optax.adam(self.config.lr)
            self.opt_state = self.optim.init(self.variables()['params'])
            self.step_count = 0
            self.key = random.PRNGKey(42)

        def belief_dist(self, k):
            return Dist.normal(self.belief_means[k], jnp.exp(0.5*self.belief_logvars[k])+1e-4)

        def goal_dist(self, k):
            return Dist.normal(self.goal_means[k], jnp.exp(0.5*self.goal_logvars[k])+1e-4)

        def policy_dist(self, k, z):
            h = self.policy_nets[k](z)
            mu_a, logvar_a = jnp.split(h, 2, -1)
            return Dist.normal(mu_a, jnp.exp(0.5*logvar_a)+1e-4)

        def imagine(self, z, a, k, steps):
            """Particle imagination rollout."""
            zs = [z]
            for _ in range(steps):
                trans_dist, _ = self.levels[k](zs[-1], a, self.key)
                z_next = Dist.sample(trans_dist, self.key, 1)[0]
                zs.append(z_next)
            return jnp.stack(zs)  # (steps+1, D)

        def expected_free_energy(self, k, z, a, goal_dist):
            """G(π) = E[ -ln p(o|z) ] + KL[q(z')||p(z')]  (simplified)"""
            trans_dist, obs_dist = self.levels[k](z, a, self.key)
            # Expected observation cost: -E[ln p(o|z)] where o ~ goal_dist
            # Approximate by KL between predicted obs and goal (at level 0) or latent goal (higher levels)
            if k == 0:
                # obs_dim matches goal_dim
                pred_obs_dist = obs_dist
                goal_obs_dist = goal_dist
            else:
                pred_obs_dist = trans_dist  # next latent is "observation" for higher level
                goal_obs_dist = goal_dist
            obs_cost = Dist.kl_divergence(pred_obs_dist, goal_obs_dist).mean()
            kl_cost = Dist.kl_divergence(trans_dist, self.goal_dist(k)).mean()
            return obs_cost + self.config.kl_weight * kl_cost

        def update_beliefs(self, obs, actions):
            """Variational inference step (single gradient step on F)."""
            self.key, *subkeys = random.split(self.key, len(self.config.latent_dims)+1)
            params = self.variables()['params']
            def loss_fn(bel_means, bel_logvars):
                total_F = 0.0
                for k in range(len(self.config.latent_dims)):
                    q = Dist.normal(bel_means[k], jnp.exp(0.5*bel_logvars[k])+1e-4)
                    # Prior from previous level (or transition from previous step)
                    if k == 0:
                        # p(z|o) ∝ p(o|z)p(z)
                        _, obs_dist = self.levels[0](bel_means[0], actions[0], subkeys[0])
                        obs_cost = -obs_dist.log_prob(obs).mean()
                        prior_dist = self.goal_dist(0)  # placeholder
                    else:
                        prior_dist = self.goal_dist(k)  # simplified
                    kl = Dist.kl_divergence(q, prior_dist).mean()
                    total_F += obs_cost + self.config.kl_weight * kl
                return total_F
            # Gradient step on belief params
            grads = grad(loss_fn)(self.belief_means, self.belief_logvars)
            # ... apply grads to belief params (omitted for brevity)
            pass

        def act(self, obs):
            self.key, key = random.split(self.key)
            # Bottom-up inference (simplified)
            z = obs
            beliefs = []
            for k in range(len(self.config.latent_dims)):
                q = self.belief_dist(k)
                z = q.mode()
                beliefs.append(z)
            # Top-down policy selection (simplified: greedy on G)
            actions = []
            for k in reversed(range(len(self.config.latent_dims))):
                pi_dist = self.policy_dist(k, beliefs[k])
                a = pi_dist.mode()
                actions.append(a)
            return actions[0]  # lowest level action

        def conciousing_state(self, obs, action) -> Dict[str, Any]:
            """Compute C_t = (b_t, π_t, G_t, P_t, ι_t)"""
            b_ents = [Dist.entropy(self.belief_dist(k)).sum() for k in range(len(self.config.latent_dims))]
            pi_ents = []
            sim_horizons = []
            iotas = []
            for k in range(len(self.config.latent_dims)):
                z = self.belief_means[k]
                pi = self.policy_dist(k, z)
                pi_ents.append(Dist.entropy(pi).sum())
                # Simulation horizon
                H = self.config.timescales[k] * (k+1)
                sim_horizons.append(H)
                # Integration (Free Energy Gap)
                F_prior = self.expected_free_energy(k, z, pi.mode(), self.goal_dist(k))
                # Posterior free energy (after update)
                F_post = F_prior - 0.1  # placeholder
                iotas.append(F_prior - F_post)
            # Aggregate
            total_iota = sum(iotas)
            surprise = -self.levels[0](self.belief_means[0], action, self.key)[1].log_prob(obs).sum()
            return {
                "t": self.step_count,
                "wall_time": time.time(),
                "belief_entropy": float(sum(b_ents)),
                "policy_entropy": float(sum(pi_ents)),
                "goal_stack_depth": len(self.config.latent_dims),
                "sim_horizon": sum(sim_horizons),
                "integration_proxy": float(total_iota),
                "surprise": float(surprise)
            }

        def step(self, env_obs, env_action, env_reward, done):
            cs = self.conciousing_state(env_obs, env_action)
            self.step_count += 1
            return cs
else:
    # PyTorch version (similar structure)
    class HAIAgent(nn.Module):
        def __init__(self, config: HAIConfig):
            super().__init__()
            self.config = config
            self.levels = nn.ModuleList([LevelWorldModel(
                latent_dim=d, obs_dim=config.obs_dim if k==0 else config.latent_dims[k-1],
                action_dim=config.action_dim if k==0 else config.latent_dims[k-1],
                hidden=config.transition_hidden
            ) for k,d in enumerate(config.latent_dims)])
            self.policy_nets = nn.ModuleList([MLP([d, config.policy_hidden, config.policy_hidden, config.action_dim*2])
                                              for d in config.latent_dims])
            self.register_parameter("belief_means", nn.ParameterList([
                nn.Parameter(torch.zeros(d)) for d in config.latent_dims
            ]))
            self.register_parameter("belief_logvars", nn.ParameterList([
                nn.Parameter(torch.zeros(d)) for d in config.latent_dims
            ]))
            self.register_parameter("goal_means", nn.ParameterList([
                nn.Parameter(torch.zeros(d)) for d in config.latent_dims
            ]))
            self.register_parameter("goal_logvars", nn.ParameterList([
                nn.Parameter(torch.zeros(d)) for d in config.latent_dims
            ]))
            self.optim = torch.optim.Adam(self.parameters(), lr=config.lr)
            self.step_count = 0
        # ... (similar methods, omitted for brevity)
        pass

# ===== Conciousing Probe (Pluggable) =====
@dataclass
class ConciousingState:
    t: int
    wall_time: float
    belief_entropy: float
    policy_entropy: float
    goal_stack_depth: int
    sim_horizon: int
    integration_proxy: float
    surprise: float

class ConciousingProbe:
    def __init__(self, agent: HAIAgent, estimator: str = "FE"):
        self.agent = agent
        self.estimator = estimator
        self.log: List[ConciousingState] = []
    def step(self, obs, action) -> ConciousingState:
        cs_dict = self.agent.conciousing_state(obs, action)
        cs = ConciousingState(**cs_dict)
        self.log.append(cs)
        return cs
    def save_jsonl(self, path: str):
        with open(path, 'w') as f:
            for cs in self.log:
                f.write(json.dumps(asdict(cs)) + '\n')

# ===== Minimal Environment (CartPole-like) =====
class SimpleEnv:
    def __init__(self, obs_dim=4, action_dim=2):
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.state = np.zeros(obs_dim)
    def reset(self):
        self.state = np.random.randn(self.obs_dim) * 0.1
        return self.state.astype(np.float32)
    def step(self, action):
        # Random walk dynamics for demo
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.obs_dim) + 0.01 * action
        reward = -np.sum(self.state**2)  # want to stay near origin
        done = False
        return self.state.astype(np.float32), reward, done, {}

# ===== Main Training Loop =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["jax","torch"], default=BACKEND)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--profile-out", type=str, default="conciousing_profile.jsonl")
    parser.add_argument("--live-plot", action="store_true")
    args = parser.parse_args()

    config = HAIConfig()
    agent = HAIAgent(config)
    probe = ConciousingProbe(agent, estimator=config.estimator)
    env = SimpleEnv(config.obs_dim, config.action_dim)

    obs = env.reset()
    if BACKEND == "jax":
        obs = jnp.array(obs)

    print(f"[{BACKEND.upper()}] Starting HAI agent conciousing profile...")
    start = time.time()
    for step in range(args.steps):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(np.array(action) if BACKEND=="jax" else action.detach().numpy())
        cs = probe.step(obs, action)
        if step % 100 == 0:
            print(f"  t={cs.t:4d}  H[b]={cs.belief_entropy:6.2f}  H[π]={cs.policy_entropy:6.2f}  "
                  f"ι={cs.integration_proxy:6.3f}  surprise={cs.surprise:6.3f}  sim_H={cs.sim_horizon}")
        obs = jnp.array(next_obs) if BACKEND=="jax" else torch.tensor(next_obs)
        if done:
            obs = jnp.array(env.reset()) if BACKEND=="jax" else torch.tensor(env.reset())
    probe.save_jsonl(args.profile_out)
    print(f"\nSaved {len(probe.log)} conciousing states to {args.profile_out}")
    print(f"Total time: {time.time()-start:.1f}s")

    if args.live_plot:
        plot_profile(args.profile_out)

def plot_profile(path):
    import matplotlib.pyplot as plt
    data = [json.loads(l) for l in open(path)]
    ts = [d['t'] for d in data]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    keys = ['belief_entropy','policy_entropy','integration_proxy','surprise','goal_stack_depth','sim_horizon']
    for ax, k in zip(axes.flat, keys):
        ax.plot(ts, [d[k] for d in data], lw=0.8)
        ax.set_ylabel(k)
    axes[-1,0].set_xlabel('t'); axes[-1,1].set_xlabel('t')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()