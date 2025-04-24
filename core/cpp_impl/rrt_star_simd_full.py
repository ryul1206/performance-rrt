#! /usr/bin/env python3

from __future__ import annotations
from typing import Tuple

from core.cpp_impl.rrt_star_simd_full_bind import rrt_star_simd
from core.rrt_star import make_description, run
from core.utils import SAMPLE_OBSTACLES, Params
from core.logger import Logger


IMPL_TYPE = "C++ SIMD Implementation"
DESCRIPTION = make_description(IMPL_TYPE)

Point2D = Tuple[float, float]
Line2D = Tuple[Point2D, Point2D]


def rrt(params: Params, logger: Logger) -> list[Line2D] | None:
    # Unpack parameters
    start: Point2D = params.start_point
    end: Point2D = params.end_point
    max_step_size: float = params.max_step_size
    neighborhood_size: float = params.neighborhood_size
    max_iter: int | None = params.max_iterations

    # Logger
    logger.log_obstacles(params.wall_segments)

    # Run RRT* algorithm with SIMD optimization
    path = rrt_star_simd(
        start,
        end,
        params.wall_segments,
        max_step_size,
        neighborhood_size,
        max_iter if max_iter is not None else 1000
    )

    # Convert path to line segments
    if path:
        segments = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        logger.log_path(segments)
        return segments
    return None


def main() -> None:
    params = Params(
        max_iterations=1000,
        max_step_size=0.1,
        start_point=(0.2, 0.5),
        end_point=(1.8, 0.5),
        wall_segments=SAMPLE_OBSTACLES,
    )
    run(IMPL_TYPE, DESCRIPTION, params, rrt)


if __name__ == "__main__":
    main()
