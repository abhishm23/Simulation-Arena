"""Discrete-event simulation engine.

Models a process with queues: deals arriving, waiting for a finite number of
approvers, moving through stages, occasionally stalling. Unlike Monte Carlo
(which asks "what is the distribution of the answer?"), DES asks "what happens
when things pile up?" - which is exactly the question for approval bottlenecks.

Stages are sequential and FIFO; each stage may draw on a shared resource pool.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import distributions as dist
from . import stats as st


def run(
    n_entities: int,
    stages: Sequence[dict],
    resources: dict[str, int] | None = None,
    arrival_spec: dict | None = None,
    rng: np.random.Generator | None = None,
    horizon: float | None = None,
    renege_after: float | None = None,
    renege_prob: float = 0.0,
) -> dict:
    """Simulate a queueing process.

    stages: [{"name", "dist": <distribution spec>, "resource": "cf", "capacity": 2}]
    arrival_spec: distribution spec for inter-arrival time (default: exponential)
    Returns per-entity records plus stage-level and system-level statistics.
    """
    rng = rng or np.random.default_rng(0)
    resources = dict(resources or {})

    # server pools -------------------------------------------------------
    pools: dict[str, list[float]] = {}
    for s in stages:
        key = s.get("resource") or s.get("name")
        cap = int(s.get("capacity", resources.get(key, 1)))
        pools[s["name"]] = [0.0] * max(cap, 1)

    arrivals: list[float] = []
    t = 0.0
    a_spec = arrival_spec or {"dist": "exponential", "rate": 1.0}
    for _ in range(n_entities):
        t += float(dist.sample(a_spec, 1, rng)[0])
        arrivals.append(t)
    if horizon is not None:
        arrivals = [a for a in arrivals if a <= horizon]

    records: list[dict] = []
    stage_wait: dict[str, list[float]] = {s["name"]: [] for s in stages}
    stage_service: dict[str, list[float]] = {s["name"]: [] for s in stages}
    stage_busy: dict[str, float] = {s["name"]: 0.0 for s in stages}
    stage_arrivals: dict[str, list[float]] = {s["name"]: [] for s in stages}
    stage_departures: dict[str, list[float]] = {s["name"]: [] for s in stages}

    for eid, arrival in enumerate(arrivals):
        clock = arrival
        abandoned = False
        for s in stages:
            name = s["name"]
            stage_arrivals[name].append(clock)
            pool = pools[name]
            free_at = min(pool)
            start = max(clock, free_at)
            wait = start - clock
            stage_wait[name].append(wait)

            if renege_after is not None and wait > renege_after:
                if rng.random() < renege_prob:
                    abandoned = True
                    records.append({
                        "entity": eid, "arrival": arrival, "exit": start,
                        "cycle_time": start - arrival, "total_wait": start - arrival,
                        "abandoned": True, "stage_reached": name,
                    })
                    break

            service = float(dist.sample(s.get("dist", {"dist": "constant", "value": 1.0}), 1, rng)[0])
            service = max(service, 0.0)
            stage_service[name].append(service)
            stage_busy[name] += service
            j = int(np.argmin(pool))
            pool[j] = start + service
            stage_departures[name].append(start + service)
            clock = start + service

        if not abandoned:
            records.append({
                "entity": eid, "arrival": arrival, "exit": clock,
                "cycle_time": clock - arrival,
                "total_wait": float(sum(stage_wait[s["name"]][-1] for s in stages)),
                "abandoned": False, "stage_reached": stages[-1]["name"] if stages else "",
            })

    cycle = [r["cycle_time"] for r in records]
    waits = [r["total_wait"] for r in records]
    abandon_rate = float(np.mean([r["abandoned"] for r in records])) if records else 0.0

    stage_stats = []
    for s in stages:
        n = s["name"]
        cap = len(pools[n])
        span = max(max(stage_departures[n]) if stage_departures[n] else 0.0, 1e-9)
        stage_stats.append({
            "stage": n,
            "capacity": cap,
            "n_through": len(stage_service[n]),
            "avg_wait": float(np.mean(stage_wait[n])) if stage_wait[n] else 0.0,
            "p90_wait": float(np.percentile(stage_wait[n], 90)) if stage_wait[n] else 0.0,
            "max_wait": float(np.max(stage_wait[n])) if stage_wait[n] else 0.0,
            "avg_service": float(np.mean(stage_service[n])) if stage_service[n] else 0.0,
            "utilisation": float(stage_busy[n] / (span * cap)) if span > 0 else 0.0,
        })

    return {
        "records": records,
        "stage_stats": stage_stats,
        "cycle_time_summary": st.summarize(cycle),
        "wait_summary": st.summarize(waits),
        "throughput": len([r for r in records if not r["abandoned"]]),
        "abandon_rate": abandon_rate,
        "makespan": float(max((r["exit"] for r in records), default=0.0)),
        "arrivals": arrivals,
    }


def prob_complete_within(result: dict, limit: float) -> float:
    rec = result["records"]
    if not rec:
        return 0.0
    done = [r for r in rec if not r["abandoned"] and r["cycle_time"] <= limit]
    return len(done) / len(rec)


def backlog_series(result: dict, horizon: float, bins: int = 40) -> tuple[list[float], list[float]]:
    """Open items over time (arrivals minus completions)."""
    arr = np.asarray(result["arrivals"], dtype=float)
    dep = np.asarray([r["exit"] for r in result["records"] if not r["abandoned"]], dtype=float)
    if arr.size == 0:
        return [], []
    edges = np.linspace(0, max(horizon, arr.max() * 1.01), bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    opened = np.histogram(arr, bins=edges)[0].cumsum()
    closed = np.histogram(dep, bins=edges)[0].cumsum()
    return centres.tolist(), (opened - closed).tolist()
