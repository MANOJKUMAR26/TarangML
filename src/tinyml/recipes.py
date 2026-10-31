"""Recipe files: per-use-case defaults so the user never sees ML terms."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel

from tinyml.config import recipes_dir


class Recipe(BaseModel):
    id: str
    version: int
    title: str
    summary: str
    keywords: list[str]
    default_task: str
    sensor: str
    sensor_ma: float = 0.2
    window_ms: float
    hop_ratio: float
    max_window_samples: int
    include_magnitude: bool
    feature_families: list[str]
    min_minutes: dict[str, float]
    min_windows_per_class: int
    targets: dict[str, dict[str, float]]
    model_budget_kb: float
    starting_rung: str
    default_max_latency_ms: float
    normal_label_hints: list[str]

    def target(self, task: str) -> dict[str, float]:
        return self.targets[task]


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, Recipe]:
    recipes: dict[str, Recipe] = {}
    for path in sorted(recipes_dir().glob("*.yaml")):
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        recipe = Recipe(**data)
        recipes[recipe.id] = recipe
    return recipes


def get_recipe(recipe_id: str) -> Recipe:
    recipes = load_recipes()
    if recipe_id not in recipes:
        raise KeyError(f"Unknown recipe '{recipe_id}'. Known: {', '.join(recipes)}")
    return recipes[recipe_id]
