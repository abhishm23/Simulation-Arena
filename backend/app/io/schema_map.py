"""Column -> role mapping.

A use case never asks for "column G". It asks for *roles* - 'deal value',
'cost to deliver', 'discount given'. This module guesses which of the user's
columns plays which role, explains the guess in plain English, and then builds
role-keyed rows for the engine.

IMPORTANT (for the backend agent)
---------------------------------
    rows, issues = apply_mapping(rows, mapping, roles)

`apply_mapping` returns a **tuple** `(rows, issues)`, NOT a bare list. The
briefing shows it returning a list; this is the one deliberate deviation. Rows
are keyed by ROLE NAME (never the raw column name), `issues` is a list of
plain-English warnings you can append to RunResult.warnings.

`mapping` values may be:
    "ColumnName"        -> take that column from each row
    {"const": 42}       -> the same value for every row
    None / absent       -> RoleSpec.fallback (or None)

Correctness contract (rule 8): *the mapping the user saw must be the mapping
actually used.* This module is the single source of truth for that:

    mapping, confidence, notes, was_auto = resolve_mapping(
        role_mapping, columns, profiles, roles, n_rows=profile.n_rows)
    mapped, issues = apply_mapping(rows, mapping, roles)
    result.assumptions += describe_mapping(mapping, roles)

`resolve_mapping` never overrides a user choice and never re-derives one; when
the user has chosen nothing it falls back to `auto_map`. `describe_mapping`
turns the mapping into plain-English lines naming every role and the column
actually used - put them in the run log and in `result.assumptions`.

Dtype compatibility
-------------------
Guessing a column is not a string game. A role has a *kind of value* it needs
(numeric, categorical, identifier, boolean, datetime) and the column has a kind
of value it actually holds. `DTYPE_COMPATIBILITY` is the whole policy: one named
table of adjustments, plus `HARD_BLOCKS`, the pairs that are **impossible** for
automatic mapping (they score below `MIN_SCORE`, so they are never suggested).

A near-unique string column is an identifier in disguise, so when the row count
is known (`n_rows=`) a categorical column with more than half as many distinct
values as rows - or more than 200 of them - is treated as an identifier for the
block rules. A *strong name match* (the column is literally called what the role
is called, or an alias of it) is exempt from that heuristic: a column named
'customer' is a customer, however many distinct customers there are.

Compatibility governs **automatic** suggestions only. A mapping the user chose
is honoured unconditionally - see `resolve_mapping`.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from ..core.contracts import ColumnProfile, RoleSpec
from .ingest import coerce_value

#: below this, we would rather leave a role unmapped than guess
MIN_SCORE = 0.45

#: no name relationship at all - the pair is not even a candidate
_NO_RELATION = -1.0

#: the name relates, but the column/role dtypes make the pairing impossible.
#: Deliberately below MIN_SCORE so a hard block can never be auto-mapped.
HARD_BLOCK_SCORE = -2.0

#: a name match this strong (exact key / alias / normalised name) overrides the
#: cardinality heuristic - a column called 'customer' is a customer.
_STRONG_NAME = 0.90

#: how many distinct values make a string column "too many to be a group"
_MAX_GROUP_VALUES = 200
#: and what share of the rows is "almost one value per row"
_NEAR_UNIQUE_SHARE = 0.50

# ---------------------------------------------------------------------------
# THE dtype compatibility policy - read this to know what we will and will not
# suggest. Keyed by (column dtype, role dtype).
# ---------------------------------------------------------------------------
#: Automatic mapping is IMPOSSIBLE for these pairs. They score below MIN_SCORE
#: and are never suggested, however well the names line up.
HARD_BLOCKS: frozenset[tuple[str, str]] = frozenset({
    ("identifier", "categorical"),   # one ID per row is not a group
    ("identifier", "numeric"),       # an ID is not a number
    ("numeric", "categorical"),      # a number is not a group
    # ("categorical", "numeric") becomes this pair once the cardinality rule
    # above promotes a near-unique string column to an identifier.
})

#: The adjustment added to the *name* score for each (column dtype, role dtype)
#: pair. Positive = the types support the guess; negative = they disagree.
DTYPE_COMPATIBILITY: dict[tuple[str, str], float] = {
    # the types agree exactly
    ("numeric", "numeric"): 0.05,
    ("categorical", "categorical"): 0.05,
    ("identifier", "identifier"): 0.05,
    ("boolean", "boolean"): 0.05,
    ("datetime", "datetime"): 0.05,
    # 1/0 and yes/no are the same idea
    ("numeric", "boolean"): 0.02,
    ("boolean", "numeric"): 0.02,
    # IDs are often just numbers - no help, no harm
    ("numeric", "identifier"): 0.00,
    # a string key that repeats (a quote id across its lines) is read as a
    # category by the profiler but is still a perfectly good identifier
    ("categorical", "identifier"): 0.00,
    # a number is not a group unless it is a near-unique string column, which
    # the cardinality rule turns into an ("identifier", "numeric") hard block
    ("categorical", "numeric"): -0.25,
}

#: anything not named above really disagrees (e.g. datetime vs numeric)
DEFAULT_ADJUSTMENT = -0.25

# ---------------------------------------------------------------------------
# The measure policy: money and time are not interchangeable.
# ---------------------------------------------------------------------------
# QA NEW-01: the UC5 deal-age role ("How long the deal has been sitting
# (quarters)") is a numeric role, and the leftover money column 'deal_acv_usd'
# is a numeric column, so name overlap on the generic word "deal" was enough to
# pair them - the app then reported deals as having sat 1,090,947 quarters.
# A money column is not a length of time, and a length of time is not an amount
# of money, however well the names line up: the pairing is IMPOSSIBLE for
# automatic mapping, exactly like the dtype hard blocks above.
_MONEY_WORDS: frozenset[str] = frozenset({
    "amount", "amt", "value", "val", "usd", "eur", "gbp", "dollar", "dollars",
    "acv", "tcv", "arr", "mrr", "gmv", "revenue", "rev", "price", "cost",
    "cogs", "spend", "expense", "margin", "profit", "cash", "discount",
    "rebate", "invoice", "booking", "bookings", "budget",
})

_DURATION_WORDS: frozenset[str] = frozenset({
    "age", "day", "days", "week", "weeks", "month", "months", "quarter",
    "quarters", "qtr", "year", "years", "tenure", "duration", "elapsed",
    "sitting", "stalled", "time", "cycle", "period", "since",
})


def _measure(text: Any) -> str:
    """What kind of quantity a name describes: 'money', 'duration' or ''.

    Empty when the name says nothing (or says both), so the rule only ever
    fires on a name that is unambiguously one or the other.
    """
    toks = set(_tokens(text))
    money = bool(toks & _MONEY_WORDS)
    duration = bool(toks & _DURATION_WORDS)
    if money == duration:                     # neither, or both - no opinion
        return ""
    return "money" if money else "duration"


def _measure_conflict(col_name: Any, role: RoleSpec) -> str:
    """Plain-English 'why not' when a column's measure cannot serve a role."""
    col_measure = _measure(col_name)
    role_measure = _measure(" ".join(_role_candidates(role)))
    if not col_measure or not role_measure or col_measure == role_measure:
        return ""
    if role_measure == "duration":
        return "an amount of money, not a length of time"
    return "a length of time, not an amount of money"

#: the most confidence a match of each kind is allowed to claim
CONFIDENCE_CAP: dict[str, float] = {
    "exact": 1.00,      # the names and the types both agree
    "neutral": 1.00,    # the types neither support nor contradict the guess
    "weak": 0.40,       # name only, and the types disagree - a forced guess
    "unknown": 0.60,    # name only, and we could not check the types
}

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "and", "or",
    "is", "are", "was", "per", "at", "by", "your", "given", "this", "that",
}

#: words that mean the same thing to a human but not to a string comparison
_SYNONYMS = [
    {"pct", "percent", "percentage", "pcnt", "prc", "rate"},
    {"amt", "amount"},
    {"val", "value", "values"},
    {"id", "identifier", "code", "key", "no", "num", "number", "ref"},
    {"cust", "customer", "client", "account", "acct", "company"},
    {"qty", "quantity", "units", "volume"},
    {"margin", "mar", "gm", "markup"},
    {"cost", "costs", "cogs", "spend", "expense"},
    {"discount", "disc", "rebate", "concession", "off"},
    {"price", "listprice", "asp", "tariff", "rate"},
    {"won", "win", "wins", "closedwon", "outcome", "success"},
    {"stage", "phase", "step", "status"},
    {"approver", "owner", "reviewer", "agent", "resource"},
    {"supplier", "vendor", "provider", "incumbent"},
    {"year", "yr", "period", "fy"},
    {"quarter", "qtr", "q"},
    {"region", "territory", "geo", "area", "market"},
    {"segment", "tier", "band", "cohort", "bucket", "category"},
]
_CANON: dict[str, str] = {}
for _group in _SYNONYMS:
    for _w in _group:
        _CANON[_w] = sorted(_group)[0]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def _norm(s: Any) -> str:
    """Lowercase, keep letters and digits only: 'Gross Margin %' -> 'grossmargin'."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _tokens(s: Any) -> list[str]:
    """Split a name into words, camelCase-aware: 'marginPct' -> ['margin','pct']."""
    t = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s or ""))
    t = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", t)
    out = []
    for part in re.split(r"[^A-Za-z0-9]+", t):
        p = part.strip().lower()
        if not p or p in _STOPWORDS:
            continue
        out.append(p)
    return out


def _canon_token(t: str) -> str:
    return _CANON.get(t, t)


#: words too generic to count as evidence on their own. Every pipeline file has
#: a 'deal' column, so a shared 'deal' must not be enough to pair a column with
#: a role - that is how the money column 'deal_fiscal_quarter_end_offset' was
#: offered as "how long the deal has been sitting" (QA NEW-01).
_WEAK_TOKENS: frozenset[str] = frozenset({
    "deal", "value", "values", "number", "num", "count", "total", "name",
    "flag", "id", "code", "key", "row", "record", "line", "item", "data",
})
_WEAK_CANON: frozenset[str] = frozenset(_canon_token(w) for w in _WEAK_TOKENS)


def _evidence_tokens(tokens: set[str]) -> set[str]:
    """Drop the generic words, unless that would leave nothing to go on."""
    strong = tokens - _WEAK_CANON
    return strong or tokens


def _role_candidates(role: RoleSpec) -> list[str]:
    out = [role.key]
    out.extend(role.aliases or [])
    if role.label:
        out.append(role.label)
    return [c for c in out if c]


def _role_tokens(role: RoleSpec) -> set[str]:
    toks: set[str] = set()
    for cand in (role.key, role.label):
        toks.update(_tokens(cand))
    for alias in (role.aliases or []):
        toks.update(_tokens(alias))
    return _evidence_tokens({_canon_token(t) for t in toks if t})


def _token_score(col_tokens: Sequence[str], role_tokens: set[str]) -> float:
    """Shared-word score in the 0.70 - 0.85 band, as specified."""
    cs = _evidence_tokens({_canon_token(t) for t in col_tokens})
    if not cs or not role_tokens:
        return 0.0
    inter = cs & role_tokens
    if not inter:
        return 0.0
    ratio = len(inter) / max(1, min(len(cs), len(role_tokens)))
    return 0.70 + 0.15 * min(ratio, 1.0)


def _substring_score(col_norm: str, cand_norm: str) -> float:
    if not col_norm or not cand_norm or col_norm == cand_norm:
        return 0.0
    if cand_norm in col_norm or col_norm in cand_norm:
        short, long = sorted((col_norm, cand_norm), key=len)
        return 0.72 + 0.10 * (len(short) / len(long))
    return 0.0


# ---------------------------------------------------------------------------
# Name evidence: how much does the column's *name* look like the role?
# ---------------------------------------------------------------------------
def _on_word_boundary(column: Any, cand: Any) -> bool:
    """Does `cand` sit on a word boundary inside `column`?

    A substring hit *inside* a longer word is a false positive: 'age' is
    contained in 'eng-age-ment', which is how 'stakeholder_engagement_velocity'
    scored as a deal-age column (QA NEW-01). Word-level alignment keeps
    'gross_margin' -> 'margin' while rejecting 'engagement' -> 'age'.

    Names that do not tokenise (a single run of letters, e.g. 'marginpct') keep
    the old plain-substring behaviour - there are no boundaries to align on.
    """
    col_tokens = [_canon_token(t) for t in _tokens(column)]
    cand_tokens = [_canon_token(t) for t in _tokens(cand)]
    if len(col_tokens) <= 1 or not cand_tokens:
        return True
    n = len(cand_tokens)
    return any(col_tokens[i:i + n] == cand_tokens
               for i in range(len(col_tokens) - n + 1))


def _name_score(column: str, role: RoleSpec) -> float:
    """How strongly the column's name matches the role, 0.0 = no relationship.

    This is name evidence only - it says nothing about whether the column's
    *contents* can play the role. `_score_pair` combines the two.
    """
    col_norm = _norm(column)
    base = 0.0
    for cand in _role_candidates(role):
        cand_norm = _norm(cand)
        if not cand_norm:
            continue
        if str(column).strip().lower() == str(cand).strip().lower():
            base = max(base, 1.00)                       # exact key match
        elif col_norm == cand_norm:
            base = max(base, 0.95 if cand == role.key else 0.90)   # normalised / alias
    for cand in (role.aliases or []):
        cand_norm = _norm(cand)
        if cand_norm and cand_norm in col_norm and col_norm != cand_norm:
            if _on_word_boundary(column, cand):
                base = max(base, _substring_score(col_norm, cand_norm))

    base = max(base, _token_score(_tokens(column), _role_tokens(role)))
    if _on_word_boundary(column, role.key):
        base = max(base, _substring_score(col_norm, _norm(role.key)))
    return base


# ---------------------------------------------------------------------------
# Dtype evidence: can a column holding `col_dtype` play a `role_dtype` role?
# ---------------------------------------------------------------------------
def _near_unique(prof: Optional[ColumnProfile], n_rows: Optional[int]) -> bool:
    """Is this column effectively one value per row (an identifier in disguise)?

    Uses `ColumnProfile.unique` and the dataset row count. Without the row
    count we cannot judge, so the rule is skipped (`n_rows=None`).
    """
    if prof is None or n_rows is None:
        return False
    try:
        n = int(n_rows)
        u = int(prof.unique or 0)
    except Exception:
        return False
    if n <= 0 or u <= 0:
        return False
    if u > _MAX_GROUP_VALUES:
        return True
    return (u / float(n)) > _NEAR_UNIQUE_SHARE


def _block_reason(col_dtype: str, role_dtype: str, near_unique: bool) -> str:
    """Plain-English 'why not' for a rejected column, e.g. 'an ID rather than a group'."""
    if col_dtype == "identifier":
        what = ("a different value in almost every row, so it is an ID"
                if near_unique else "an ID")
        return f"{what} rather than a number" if role_dtype == "numeric" \
            else f"{what} rather than a group"
    if col_dtype == "numeric":
        return "numbers rather than groups"
    return "not something this role can use"


def _pair_policy(col_dtype: Any, role_dtype: Any, prof: Optional[ColumnProfile],
                 n_rows: Optional[int], strong_name: bool = False,
                 col_name: Any = "", role: Optional[RoleSpec] = None
                 ) -> tuple[str, float, str]:
    """The compatibility verdict for one (column dtype, role dtype) pair.

    Returns (kind, adjustment, reason) where kind is one of:
        "exact"   - the types support the guess (rewarded)
        "neutral" - allowed; the types neither support nor contradict it
        "weak"    - allowed, but the types disagree (a forced guess)
        "blocked" - impossible; never auto-mapped
        "unknown" - one side has no dtype, so we cannot judge
    `reason` is plain English and only set for "blocked".
    """
    cd = str(col_dtype or "").lower()
    rd = str(role_dtype or "").lower()
    if not cd or not rd:
        return "unknown", 0.0, ""

    # Money and time are different quantities (NEW-01): the numeric dtype alone
    # cannot tell a $40,000,000 deal from a deal 40,000,000 quarters old.
    if rd == "numeric" and role is not None:
        conflict = _measure_conflict(col_name, role)
        if conflict:
            return "blocked", HARD_BLOCK_SCORE, conflict

    near = cd == "categorical" and _near_unique(prof, n_rows)
    if near and not strong_name:
        # a near-unique string column is an identifier in disguise
        cd = "identifier"

    # A role that needs numbers can only take a column that holds numbers.
    # Without this, a category column (e.g. "segment" = SMB / Enterprise) can be
    # auto-mapped to a numeric role and then fail on every value, filling the
    # result with "we left those blank" warnings for data we never needed.
    if rd == "numeric" and not (
        cd == "numeric" or (prof is not None and prof.numeric_stats is not None)
    ):
        return "blocked", HARD_BLOCK_SCORE, "that column does not contain numbers"

    pair = (cd, rd)
    if pair in HARD_BLOCKS:
        return "blocked", HARD_BLOCK_SCORE, _block_reason(cd, rd, near)

    adj = DTYPE_COMPATIBILITY.get(pair, DEFAULT_ADJUSTMENT)
    if adj >= 0.02:
        kind = "exact"
    elif adj >= 0.0:
        kind = "neutral"
    else:
        kind = "weak"
    return kind, adj, ""


# ---------------------------------------------------------------------------
# Scoring one (column, role) pair
# ---------------------------------------------------------------------------
def _score_pair(column: str, prof: Optional[ColumnProfile], role: RoleSpec,
                n_rows: Optional[int] = None) -> float:
    """Score one (column, role) pair, 0..1, or a sentinel below MIN_SCORE.

    Combines name evidence with dtype compatibility. A hard block (see
    `HARD_BLOCKS`) returns `HARD_BLOCK_SCORE` however good the name is, so an
    implausible pairing is impossible rather than merely unlikely.
    """
    name = _name_score(column, role)
    if name <= 0.0:
        return _NO_RELATION                              # no relationship at all
    # The measure rule is about the *names*, so it applies even with no profile.
    if (role.dtype or "").lower() == "numeric" and _measure_conflict(column, role):
        return HARD_BLOCK_SCORE
    if prof is None:
        return round(min(name, 1.0), 4)                  # name only

    strong = name >= _STRONG_NAME
    kind, adj, _reason = _pair_policy(prof.dtype, role.dtype, prof, n_rows, strong,
                                      col_name=column, role=role)
    if kind == "blocked":
        return HARD_BLOCK_SCORE

    base = name + adj
    col_dtype = (prof.dtype or "").lower()
    if (role.dtype or "").lower() == "numeric" and (
            col_dtype == "numeric" or prof.numeric_stats is not None):
        base += 0.05                                     # numeric content on a numeric role
    return round(min(base, 1.0), 4)


# ---------------------------------------------------------------------------
# Public: auto_map
# ---------------------------------------------------------------------------
def auto_map(columns: Sequence[str],
             profiles: Sequence[ColumnProfile],
             roles: Sequence[RoleSpec],
             n_rows: Optional[int] = None) -> tuple[dict[str, Optional[str]],
                                                    dict[str, float],
                                                    list[str]]:
    """Guess which column plays which role.

    Returns (mapping role->column|None, confidence role->0..1, notes[str]).
    Every (column, role) pair is scored; the best pairs win, one column per
    role and one role per column. Anything scoring below 0.45 is left unmapped
    and explained in `notes`.

    `n_rows` is the dataset's row count, if known. Supplying it enables the
    cardinality rule (a near-unique string column is treated as an identifier);
    `None` skips that rule, since `ColumnProfile` does not carry the row count.
    """
    columns = list(columns or [])
    roles = list(roles or [])
    by_name = {p.name: p for p in (profiles or [])}

    pairs: list[tuple[float, int, int]] = []
    best_for_role: dict[int, list[tuple[float, str]]] = {}
    blocked_for_role: dict[int, list[tuple[float, str, str]]] = {}
    for ri, role in enumerate(roles):
        best_for_role[ri] = []
        blocked_for_role[ri] = []
        for ci, col in enumerate(columns):
            prof = by_name.get(col)
            s = _score_pair(col, prof, role, n_rows)
            if s >= MIN_SCORE:
                pairs.append((s, ri, ci))
                best_for_role[ri].append((s, col))
            elif s == HARD_BLOCK_SCORE:
                # the name looked plausible but the types rule it out - remember
                # it so we can tell the user exactly which column we rejected.
                nm = _name_score(col, role)
                if nm > 0.0:
                    reason = ""
                    if (role.dtype or "").lower() == "numeric":
                        reason = _measure_conflict(col, role)
                    if not reason:
                        _kind, _adj, reason = _pair_policy(
                            prof.dtype if prof else "", role.dtype, prof, n_rows,
                            nm >= _STRONG_NAME, col_name=col, role=role)
                    blocked_for_role[ri].append((nm, col, reason))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    mapping: dict[str, Optional[str]] = {r.key: None for r in roles}
    confidence: dict[str, float] = {r.key: 0.0 for r in roles}
    chosen_kind: dict[int, str] = {}
    used_cols: set[str] = set()
    for s, ri, ci in pairs:
        role = roles[ri]
        if mapping.get(role.key) is not None:
            continue
        col = columns[ci]
        if col in used_cols:
            continue
        prof = by_name.get(col)
        if prof is None:
            kind = "unknown"
        else:
            kind, _adj, _reason = _pair_policy(
                prof.dtype, role.dtype, prof, n_rows,
                _name_score(col, role) >= _STRONG_NAME, col_name=col, role=role)
        cap = CONFIDENCE_CAP.get(kind, CONFIDENCE_CAP["unknown"])
        mapping[role.key] = col
        confidence[role.key] = round(float(min(max(s, 0.0), cap)), 2)
        chosen_kind[ri] = kind
        used_cols.add(col)

    # -- plain-English notes ------------------------------------------------
    notes: list[str] = []

    # QA UX-03 / TM-23 / TM-24: the app silently ignored 13-15 of the user's
    # columns and never said so. Name them, so the user can see what was left
    # out and why. (Capped so a 200-column file cannot drown the panel.)
    unused = [c for c in columns if c not in used_cols]
    if unused:
        shown = ", ".join(f"'{c}'" for c in unused[:12])
        more = len(unused) - 12
        notes.append(
            f"We are not using {len(unused)} of your columns in this simulation"
            + (f" ({shown}" + (f" and {more} more)" if more > 0 else ")"))
            + ". That is normal - most files carry columns we do not need - but "
              "if one of them matters to you, tell us which role it plays.")

    for ri, role in enumerate(roles):
        col = mapping.get(role.key)
        if col:
            c = confidence.get(role.key, 0.0)
            kind = chosen_kind.get(ri, "unknown")
            runners = sorted(best_for_role.get(ri, []), reverse=True)
            rival = next(((s, name) for s, name in runners if name != col), None)
            if c >= 0.90:
                lead = f"Matched '{role.label}' to your '{col}' column"
            elif c >= 0.65:
                lead = (f"We matched '{role.label}' to your '{col}' column "
                        f"- worth a quick look to be sure")
            else:
                lead = (f"Best guess only: we matched '{role.label}' to '{col}' "
                        f"- please check this one")
            notes.append(lead + _evidence_phrase(kind) + ".")
            if rival is not None and c < 0.90 and rival[0] >= c - 0.05:
                notes.append(f"'{col}' and '{rival[1]}' both look like '{role.label}'; "
                             f"we picked '{col}'.")
        else:
            blocked = sorted(blocked_for_role.get(ri, []), reverse=True)
            if blocked:
                _nm, bcol, reason = blocked[0]
                explain = (f"We could not find a '{role.label}' column - your '{bcol}' "
                           f"column is {reason}, so we did not use it as '{role.label}'.")
            else:
                explain = f"We could not find a '{role.label}' column."
            notes.append(explain + _unmapped_tail(role))
    return mapping, confidence, notes


def _evidence_phrase(kind: str) -> str:
    """What evidence drove a match, in plain English."""
    if kind == "exact":
        return " (name and type both agree)"
    if kind == "neutral":
        return " (the name fits, and the column type is close enough)"
    if kind == "weak":
        return " (name only - the column type is a poor fit, so please check)"
    return " (name only - we could not check the column type)"


def _unmapped_tail(role: RoleSpec) -> str:
    """What we do instead when a role has no column, in plain English."""
    fb = role.fallback
    if fb is not None and not isinstance(fb, dict):
        shown = fb if not isinstance(fb, float) else round(fb, 4)
        return f" We are using your standard value of {shown} instead."
    if role.required:
        return " Please choose one: this simulation needs it."
    return " We left it out and will work without it."


# ---------------------------------------------------------------------------
# Public: resolve_mapping - the single source of truth for the used mapping
# ---------------------------------------------------------------------------
def resolve_mapping(role_mapping: Optional[dict[str, Any]],
                    columns: Sequence[str],
                    profiles: Sequence[ColumnProfile],
                    roles: Sequence[RoleSpec],
                    n_rows: Optional[int] = None
                    ) -> tuple[dict[str, Any], dict[str, float], list[str], bool]:
    """The single source of truth for 'what mapping will actually be used'.

    Returns (mapping, confidence, notes, was_auto).
    If role_mapping contains any real choice it is validated and returned as-is
    (was_auto=False) - user choices are never overridden, however odd they look.
    Only when it is empty (or all values are None) do we fall back to auto_map
    (was_auto=True).

    The mapping this returns is exactly what `apply_mapping` must be given; it
    is never re-derived downstream.
    """
    roles = list(roles or [])
    supplied_in = dict(role_mapping or {})
    supplied = {k: v for k, v in supplied_in.items() if v is not None}

    if supplied:
        mapping: dict[str, Any] = {}
        for role in roles:
            mapping[role.key] = supplied_in.get(role.key, None)
        for key, value in supplied_in.items():          # keep any extra keys
            if key not in mapping:
                mapping[key] = value
        confidence = {r.key: (1.0 if mapping.get(r.key) is not None else 0.0)
                      for r in roles}
        notes: list[str] = []
        for role in roles:
            value = mapping.get(role.key)
            if value is None:
                notes.append(f"We could not find a '{role.label}' column."
                             + _unmapped_tail(role))
            elif isinstance(value, str):
                notes.append(f"You chose '{value}' for '{role.label}'.")
            else:
                notes.append(f"You set '{role.label}' yourself.")
        return mapping, confidence, notes, False

    mapping, confidence, notes = auto_map(columns, profiles, roles, n_rows=n_rows)
    return mapping, confidence, notes, True


# ---------------------------------------------------------------------------
# Public: describe_mapping - the mapping, written down for the user
# ---------------------------------------------------------------------------
def describe_mapping(mapping: Optional[dict[str, Any]],
                     roles: Sequence[RoleSpec]) -> list[str]:
    """Plain-English lines naming every role and the column actually used.

    e.g. "Tier -> your 'segment' column" /
         "Cost -> not in your file; used the standard value (0.4) instead".
    Suitable for the run log and for `result.assumptions`.
    """
    mapping = dict(mapping or {})
    roles = list(roles or [])
    out: list[str] = []
    for role in roles:
        value = mapping.get(role.key, None)
        if isinstance(value, str):
            out.append(f"{role.label} -> your '{value}' column")
        elif isinstance(value, dict):
            if "const" in value:
                out.append(f"{role.label} -> the value you set ({value['const']})")
            elif "column" in value:
                out.append(f"{role.label} -> your '{value['column']}' column")
            elif "derive" in value:
                out.append(f"{role.label} -> worked out from your other columns")
            else:
                out.append(f"{role.label} -> not in your file; not used")
        elif value is None:
            out.append(f"{role.label} -> {_describe_absent(role)}")
        else:
            out.append(f"{role.label} -> the value you set ({value})")

    known = {r.key for r in roles}
    for key, value in mapping.items():                  # extra, non-role keys
        if key in known or value is None:
            continue
        if isinstance(value, str):
            out.append(f"{key} -> your '{value}' column")
        else:
            out.append(f"{key} -> the value you set ({value})")
    return out


def _describe_absent(role: RoleSpec) -> str:
    """What happened to a role we could not map, in plain English."""
    fb = role.fallback
    if isinstance(fb, dict) and "const" in fb:
        return f"not in your file; used the standard value ({fb['const']}) instead"
    if fb is not None and not isinstance(fb, dict):
        shown = fb if not isinstance(fb, float) else round(fb, 4)
        return f"not in your file; used the standard value ({shown}) instead"
    if role.required:
        return "not in your file; this simulation needs it"
    return "not in your file; not used"


# ---------------------------------------------------------------------------
# Public: apply_mapping
# ---------------------------------------------------------------------------
def _mapping_value(m: Any, row: dict, role: Optional[RoleSpec]) -> tuple[Any, bool]:
    """Resolve one role for one row. Returns (raw_value, value_came_from_a_missing_column)."""
    if m is None:
        fb = role.fallback if role is not None else None
        if isinstance(fb, dict) and "const" in fb:
            return fb["const"], False
        return fb, False
    if isinstance(m, dict):
        if "const" in m:
            return m["const"], False
        if "column" in m:
            name = m["column"]
            return row.get(name, None), name not in row
        if "derive" in m:
            return None, False
        return None, False
    if isinstance(m, str):
        if m.startswith("const:"):
            return m[6:], False
        return row.get(m, None), m not in row
    return m, False


def apply_mapping(rows: Sequence[dict],
                  mapping: dict[str, Any],
                  roles: Sequence[RoleSpec]) -> tuple[list[dict], list[str]]:
    """Rebuild rows keyed by ROLE NAME, coerced to the role's dtype.

    Give this exactly the mapping `resolve_mapping` returned - it does not
    re-derive anything, so the mapping used is the mapping that was shown.

    Returns (rows, issues). `issues` is a list of plain-English warnings about
    values that could not be understood (they become None, never an exception).
    """
    rows = list(rows or [])
    mapping = dict(mapping or {})
    roles = list(roles or [])
    spec = {r.key: r for r in roles}

    keys: list[str] = [r.key for r in roles]
    for k in mapping:
        if k not in spec:
            keys.append(k)

    issues: list[str] = []
    warned: set[tuple[str, str]] = set()
    missing_cols: set[str] = set()
    bad_values = 0

    out: list[dict] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for key in keys:
            role = spec.get(key)
            dtype = role.dtype if role else "categorical"
            label = role.label if role else key
            m = mapping.get(key, None)
            raw, missing = _mapping_value(m, row, role)
            if missing and isinstance(m, str) and m not in missing_cols:
                missing_cols.add(m)
                issues.append(f"The column '{m}' is not in this file, so '{label}' "
                              f"was left empty.")
            value = coerce_value(raw, dtype)
            if raw is not None and raw != "" and value is None:
                bad_values += 1
                shown = str(raw)[:24]
                # One warning per column, not one per distinct value: a category
                # column used to produce a separate warning for every category
                # ("SMB", "Mid-Market", "Enterprise"), which read as far more
                # damage than there was.
                tag = (key,)
                if tag not in warned and len(issues) < 25:
                    warned.add(tag)
                    source = m if isinstance(m, str) else label
                    if dtype == "numeric":
                        issues.append(f"Some '{source}' values (for example '{shown}') "
                                      f"are not numbers - we left those rows blank.")
                    elif dtype == "datetime":
                        issues.append(f"Some '{source}' values (for example '{shown}') "
                                      f"are not dates we recognise - we left those blank.")
                    elif dtype == "boolean":
                        issues.append(f"Some '{source}' values (for example '{shown}') "
                                      f"are not yes/no - we left those blank.")
                    else:
                        issues.append(f"Some '{source}' values could not be read "
                                      f"(for example '{shown}') - we left those blank.")
            rec[key] = value
        out.append(rec)
    if bad_values > len(issues):
        issues.append(f"{bad_values} cells in total could not be read. We left them blank "
                      f"rather than guess a value.")
    return out, issues
