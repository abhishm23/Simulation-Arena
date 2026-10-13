"""Registry: use cases are plug-ins.

Adding use case #17 later must mean "drop one file in use_cases/" and nothing
else. This module discovers and exposes them.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Optional

from .contracts import RunRequest, RunResult, UseCaseMeta

_REGISTRY: dict[str, "UseCase"] = {}
_LOAD_ERRORS: dict[str, str] = {}


class UseCase:
    """Base class for a use case plug-in."""

    meta: UseCaseMeta

    def run(self, req: RunRequest, ctx: "RunContext") -> RunResult:  # pragma: no cover
        raise NotImplementedError


class RunContext:
    """Everything a use case is allowed to touch."""

    def __init__(self, narrator, rng_factory, rows: list[dict], params: dict,
                 run_id: str, req: RunRequest, profile=None):
        self.narrator = narrator
        self.rng_factory = rng_factory          # fn(*tags) -> np.random.Generator
        self.rows = rows                        # mapped + cleaned records
        self.params = params                    # resolved parameters
        self.run_id = run_id
        self.request = req
        self.profile = profile

    def rng(self, *tags: str):
        return self.rng_factory(*tags)

    def say(self, msg: str, **kw):
        self.narrator.say(msg, **kw)


def register(cls: type) -> type:
    inst = cls()
    _REGISTRY[inst.meta.id] = inst
    return cls


def discover() -> None:
    """Import every module in use_cases/ so @register decorators fire."""
    import app.use_cases as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
        except Exception as exc:  # a broken plug-in must not kill the app
            _LOAD_ERRORS[mod.name] = f"{type(exc).__name__}: {exc}"


def get(use_case_id: str) -> Optional[UseCase]:
    if not _REGISTRY:
        discover()
    return _REGISTRY.get(use_case_id)


def all_use_cases() -> list[UseCase]:
    if not _REGISTRY:
        discover()
    return list(_REGISTRY.values())


def catalog() -> list[UseCaseMeta]:
    return [uc.meta for uc in all_use_cases()]


def load_errors() -> dict[str, str]:
    return dict(_LOAD_ERRORS)
