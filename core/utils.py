#! /usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass, field


@dataclass
class Params:
    """
    RRT* Parameters
    ===============

    max_iterations: int
        Maximum number of iterations to run the algorithm.

    max_step_size: float
        Maximum distance between two points in the RRT tree.

    neighborhood_size: float
        Search radius for the RRT* rewiring step.
        In the original paper, the radius is `r_n = gamma * (ln{n} / n)^(1/d)`.
        But here, we use `max_step_size * 1.5` for convenience.

    start_point, end_point: Tuple[float, float]
        The start and end points of the path.
    """

    # Algorithm parameters
    max_iterations: int | None
    max_step_size: float
    neighborhood_size: float = field(init=False)

    # Environment parameters
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    wall_segments: list[Tuple[Tuple[float, float], Tuple[float, float]]]

    def __post_init__(self):
        self.neighborhood_size = self.max_step_size * 1.5


# Wall segments defining the environment boundaries and obstacles
SAMPLE_OBSTACLES = [
    # Outer boundary walls
    ((0, 0), (0, 1)),  # Left wall
    ((0, 1), (2, 1)),  # Top wall
    ((2, 1), (2, 0)),  # Right wall
    ((2, 0), (0, 0)),  # Bottom wall
    # Inner obstacle walls
    ((1.0, 0.0), (1.0, 0.65)),  # Center vertical wall
    ((1.5, 1.0), (1.5, 0.2)),  # Right vertical wall
    ((0.4, 0.2), (0.4, 0.8)),  # Left vertical wall
]

# Complex scenario with more obstacles and larger search space
COMPLEX_OBSTACLES = [
    # Outer walls
    ((0.0, 0.0), (2.0, 0.0)),
    ((2.0, 0.0), (2.0, 1.0)),
    ((2.0, 1.0), (0.0, 1.0)),
    ((0.0, 1.0), (0.0, 0.0)),
    # Inner obstacles
    ((0.5, 0.2), (0.5, 0.8)),
    ((0.8, 0.3), (0.8, 0.7)),
    ((1.2, 0.2), (1.2, 0.8)),
    ((1.5, 0.3), (1.5, 0.7)),
    # Diagonal obstacles
    ((0.3, 0.3), (0.7, 0.7)),
    ((1.3, 0.3), (1.7, 0.7)),
    # Small obstacles
    ((0.1, 0.1), (0.2, 0.2)),
    ((1.8, 0.8), (1.9, 0.9)),
    ((0.1, 0.8), (0.2, 0.9)),
    ((1.8, 0.1), (1.9, 0.2)),
]

if __name__ == "__main__":
    print(SAMPLE_OBSTACLES)
