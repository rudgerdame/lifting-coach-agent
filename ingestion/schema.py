"""Canonical data models for workout and recovery ingestion."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class WorkoutSet(BaseModel):
    """One logged set — see docs/data-schema.md."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    session_id: str
    timestamp: datetime
    exercise: str
    muscle_group: Optional[str] = None
    set_number: int = Field(ge=1)
    reps: int = Field(ge=0)
    weight_kg: float = Field(ge=0)
    bodyweight_kg: Optional[float] = None
    is_warmup: bool = False
    notes: Optional[str] = None


class RecoveryDaily(BaseModel):
    """Daily recovery signals aggregated from Apple Health or manual CSV."""

    date: date
    sleep_hours: Optional[float] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    bodyweight_kg: Optional[float] = None
    resting_hr_bpm: Optional[float] = None
