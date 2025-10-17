from typing import Dict, Type

from exercises.squats import SquatsExercise


EXERCISE_REGISTRY: Dict[str, Type] = {
    SquatsExercise.name: SquatsExercise,
}


def get_exercise(name: str):
    key = name.strip().lower()
    if key not in EXERCISE_REGISTRY:
        raise KeyError(f"Unknown exercise: {name}")
    return EXERCISE_REGISTRY[key]()


def available_exercises():
    return sorted(EXERCISE_REGISTRY.keys())


