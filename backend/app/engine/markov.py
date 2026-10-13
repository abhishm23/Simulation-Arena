"""Discrete-time Markov chain engine for pipeline / stage-to-stage flows.

A deal is not "won or lost in Q2" - it hops between stages with probabilities
we can measure from history. This engine turns a transition matrix into:
  * where the deal is likely to be, quarter by quarter
  * the probability it eventually lands in each absorbing state (Won / Dead)
  * how long it is expected to take
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def build_matrix(states: Sequence[str],
                 transitions: Sequence[tuple[str, str, float]]) -> np.ndarray:
    """transitions: list of (from_state, to_state, weight). Rows normalised."""
    idx = {s: i for i, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for a, b, w in transitions:
        if a in idx and b in idx:
            M[idx[a], idx[b]] += float(w)
    for i in range(len(states)):
        row = M[i].sum()
        if row <= 0:
            M[i, i] = 1.0          # unknown state -> treat as sticky
        else:
            M[i] /= row
    return M


def from_event_log(records: Sequence[tuple],
                   smoothing: float = 0.5) -> tuple[list[str], np.ndarray, dict]:
    """records: [(entity_id, time_order, state), ...] sorted by time per entity.

    Returns (states, matrix, counts) where counts keeps the raw evidence so the
    UI can show "based on 412 observed stage moves".
    """
    by_entity: dict = defaultdict(list)
    for eid, t, s in records:
        by_entity[eid].append((t, s))
    counts: dict[tuple[str, str], float] = defaultdict(float)
    states: list[str] = []
    seen = set()
    for eid, seq in by_entity.items():
        seq = sorted(seq, key=lambda x: x[0])
        for (_, a), (_, b) in zip(seq, seq[1:]):
            counts[(a, b)] += 1.0
            if a not in seen:
                seen.add(a); states.append(a)
            if b not in seen:
                seen.add(b); states.append(b)
    if not states:
        return [], np.zeros((0, 0)), {}
    # additive smoothing so a state never has a zero row from thin data
    trans = [(a, b, w) for (a, b), w in counts.items()]
    for a in states:
        for b in states:
            trans.append((a, b, smoothing))
    return states, build_matrix(states, trans), dict(counts)


def classify(states: Sequence[str], M: np.ndarray) -> dict:
    absorbing = [s for i, s in enumerate(states) if M[i, i] >= 1.0 - 1e-9
                 and M[i].sum() - M[i, i] <= 1e-9]
    return {"absorbing": absorbing,
            "transient": [s for s in states if s not in absorbing]}


def forecast(M: np.ndarray, states: Sequence[str], start,
             horizon: int) -> dict:
    """start: {state: prob} or a probability vector or a single state name."""
    k = len(states)
    if isinstance(start, str):
        v = np.zeros(k); v[states.index(start)] = 1.0
    elif isinstance(start, dict):
        v = np.zeros(k)
        for s, p in start.items():
            if s in states:
                v[states.index(s)] = float(p)
        if v.sum() <= 0:
            v[0] = 1.0
        v /= v.sum()
    else:
        v = np.asarray(start, dtype=float)
        v = v / v.sum() if v.sum() > 0 else np.eye(k)[0]

    path = [v.copy()]
    for _ in range(horizon):
        v = v @ M
        path.append(v.copy())

    info = classify(states, M)
    absorption = absorption_probabilities(M, states, path[0])
    steps = expected_steps(M, states)

    return {
        "states": list(states),
        "distribution": [list(map(float, p)) for p in path],   # horizon+1 rows
        "absorbing": info["absorbing"],
        "transient": info["transient"],
        "absorption": absorption,
        "expected_steps": steps,
        "expected_steps_overall": float(
            sum(path[0][i] * steps.get(states[i], 0.0) for i in range(k))
        ),
    }


def absorption_probabilities(M: np.ndarray, states: Sequence[str],
                             start) -> dict[str, float]:
    """Probability of eventually ending in each absorbing state."""
    k = len(states)
    info = classify(states, M)
    absorbing = info["absorbing"]
    transient = info["transient"]
    if not absorbing or not transient or len(absorbing) == k:
        v = np.zeros(k)
        if isinstance(start, str) and start in states:
            v[states.index(start)] = 1.0
        return {s: float(v[states.index(s)]) for s in states}

    ai = [states.index(s) for s in absorbing]
    ti = [states.index(s) for s in transient]
    Q = M[np.ix_(ti, ti)]
    R = M[np.ix_(ti, ai)]
    try:
        N = np.linalg.inv(np.eye(len(ti)) - Q)
        B = N @ R
    except np.linalg.LinAlgError:
        B = np.zeros((len(ti), len(ai)))

    v = np.zeros(k)
    if isinstance(start, str) and start in states:
        v[states.index(start)] = 1.0
    elif isinstance(start, dict):
        for s, p in start.items():
            if s in states:
                v[states.index(s)] = float(p)
        v = v / v.sum() if v.sum() > 0 else v
    elif start is not None:
        arr = np.asarray(start, dtype=float).ravel()
        if arr.size == k:
            v = arr / arr.sum() if arr.sum() > 0 else arr
    start_t = v[ti]
    res = start_t @ B
    out = {s: 0.0 for s in states}
    for j, s in enumerate(absorbing):
        out[s] = float(res[j])
    return out


def expected_steps(M: np.ndarray, states: Sequence[str]) -> dict[str, float]:
    """Expected number of transitions before absorption, per transient state."""
    info = classify(states, M)
    absorbing, transient = info["absorbing"], info["transient"]
    if not absorbing or not transient:
        return {}
    ti = [states.index(s) for s in transient]
    Q = M[np.ix_(ti, ti)]
    try:
        N = np.linalg.inv(np.eye(len(ti)) - Q)
        t = N @ np.ones(len(ti))
    except np.linalg.LinAlgError:
        t = np.zeros(len(ti))
    return {s: float(t[j]) for j, s in enumerate(transient)}


def steady_state(M: np.ndarray, iterations: int = 2000) -> np.ndarray:
    k = M.shape[0]
    v = np.full(k, 1.0 / k)
    for _ in range(iterations):
        v = v @ M
    return v
