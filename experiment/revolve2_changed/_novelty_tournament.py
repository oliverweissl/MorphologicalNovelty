from __future__ import annotations

from random import Random
from typing import List, TypeVar


Novelty = TypeVar("Novelty", bound="SupportsLt")
Fitness = TypeVar("Fitness", bound="SupportsLt")


def novelty_tournament(rng: Random, fitnesses: List[Fitness], novelty: List[Novelty], a: float | None,  k: int) -> int:
    """
    Perform tournament selection and return the index of the best individual.

    :param rng: Random number generator.
    :param fitnesses: List of finesses of individuals that join the tournament.
    :param novelty: List of novevelties of individuals that join the tournament.
    :param a: weight of considering novetly for selecttion.
    :param k: Amount of individuals to participate in tournament.
    :returns: The index of te individual that won the tournament.
    """
    assert len(fitnesses) >= k and len(fitnesses) == len(novelty)

    participant_indices = rng.choices(population=range(len(fitnesses)), k=k)

    if a is None:
        return max(participant_indices, key=lambda i: fitnesses[i] * novelty[i])
    return max(participant_indices, key=lambda i: fitnesses[i]*(1-a) + novelty[i]*a)

