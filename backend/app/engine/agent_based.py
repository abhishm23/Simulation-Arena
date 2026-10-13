"""Agent-based modelling engine.

Generic on purpose: 500 virtual buyers, two or three competing suppliers, and
rules for how each side reacts. The engine owns the clock and the bookkeeping;
the use case owns the behavioural rules.

    agents   : list of dicts (mutable attributes)
    step_fn  : (env, agents, rng, t) -> None     mutates agents / env
    observe  : (env, agents, t) -> dict          metrics recorded each step
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from . import distributions as dist


class Agent:
    __slots__ = ("id", "attrs")

    def __init__(self, agent_id: int, attrs: dict | None = None):
        self.id = agent_id
        self.attrs: dict = dict(attrs or {})

    def __getitem__(self, k):
        return self.attrs[k]

    def __setitem__(self, k, v):
        self.attrs[k] = v

    def get(self, k, default=None):
        return self.attrs.get(k, default)

    def to_dict(self) -> dict:
        return {"id": self.id, **self.attrs}


def make_agents(n: int, attr_specs: dict[str, dict], rng: np.random.Generator,
                id_prefix: str = "A") -> list[Agent]:
    """attr_specs: {attr_name: distribution_spec or {'dist': 'choice', ...}}"""
    cols = {k: dist.sample(v, n, rng) for k, v in attr_specs.items()}
    return [Agent(i, {k: cols[k][i] for k in cols}) for i in range(n)]


def choice_agents(values: Sequence, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.asarray(values), size=n, replace=True)


def run(
    n_steps: int,
    agents: list[Agent],
    env: dict,
    step_fn: Callable[[dict, list[Agent], np.random.Generator, int], None],
    observe: Callable[[dict, list[Agent], int], dict],
    rng: np.random.Generator,
    progress: Callable[[int], None] | None = None,
) -> dict:
    history: list[dict] = []
    for t in range(n_steps):
        step_fn(env, agents, rng, t)
        history.append(observe(env, agents, t))
        if progress:
            progress(t)
    return {
        "history": history,
        "agents": agents,
        "env": env,
        "final_state": history[-1] if history else {},
    }


def compare_strategies(
    n_steps: int,
    agent_factory: Callable[[np.random.Generator], list[Agent]],
    env_factory: Callable[[], dict],
    strategies: dict[str, Callable[[dict, list[Agent], np.random.Generator, int], None]],
    observe: Callable[[dict, list[Agent], int], dict],
    rng: np.random.Generator,
    progress: Callable[[int], None] | None = None,
) -> dict[str, dict]:
    """Run the same world under each strategy with independent RNG streams."""
    out = {}
    for i, (name, step_fn) in enumerate(strategies.items()):
        sub_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        out[name] = run(n_steps, agent_factory(sub_rng), env_factory(),
                        step_fn, observe, sub_rng, progress)
    return out


def trajectory(history: Sequence[dict], key: str) -> list[float]:
    return [float(h.get(key, 0.0)) for h in history]


def endstate(history: Sequence[dict], key: str) -> float:
    return float(history[-1].get(key, 0.0)) if history else 0.0


def softmax_choice(scores: Sequence[float], rng: np.random.Generator) -> int:
    """Probabilistic supplier choice: better utility wins more often, not always."""
    s = np.asarray(scores, dtype=float)
    s = s - s.max()
    w = np.exp(s / max(float(np.std(s)) if np.std(s) > 0 else 1.0, 1e-9))
    p = w / w.sum()
    return int(rng.choice(len(s), p=p))
