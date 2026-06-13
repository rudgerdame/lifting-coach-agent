"""Infer muscle group from Fitbod exercise names when not provided in logs."""

from __future__ import annotations

# First match wins; keywords are lowercase substrings of the exercise name.
_MUSCLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("chest", ("bench", "chest", "fly", "dip", "push up", "push-up", "pec")),
    ("back", ("pull up", "pull-up", "row", "lat ", "lat pulldown", "shrug", "back")),
    ("shoulders", ("shoulder", "lateral raise", "overhead press", "delt")),
    ("biceps", ("curl", "bicep")),
    ("triceps", ("tricep", "pushdown", "push down", "skull")),
    ("quads", ("squat", "leg extension", "lunge", "split squat")),
    ("hamstrings", ("hamstring", "rdl", "romanian deadlift", "leg curl", "deadlift")),
    ("glutes", ("hip thrust", "glute", "good morning")),
    ("calves", ("calf", "calves")),
    ("core", ("leg raise", "crunch", "plank", "ab ", "abs")),
]


def infer_muscle_group(exercise: str) -> str:
    """Map a Fitbod exercise name to a coarse muscle group."""
    name = exercise.lower().strip()
    for muscle, keywords in _MUSCLE_KEYWORDS:
        if any(kw in name for kw in keywords):
            return muscle
    return "unknown"


_PUSH = frozenset({"chest", "shoulders", "triceps"})
_PULL = frozenset({"back", "biceps"})
_LEGS = frozenset({"quads", "hamstrings", "glutes", "calves"})


def infer_split(muscle_group: str) -> str:
    """Map muscle group to PPL split (known before the gym session)."""
    mg = muscle_group.lower().strip()
    if mg in _PUSH:
        return "push"
    if mg in _PULL:
        return "pull"
    if mg in _LEGS:
        return "legs"
    return "unknown"
