#!/usr/bin/env python3

import cProfile
import pstats
from core.utils import Params, SAMPLE_OBSTACLES
from core.logger import NullLogger

from core.python_impl.rrt_star_pure import rrt as rrt_pure_python

params = Params(
    max_iterations=5000,
    max_step_size=0.1,
    start_point=(0.2, 0.5),
    end_point=(1.8, 0.5),
    wall_segments=SAMPLE_OBSTACLES,
    # wall_segments=COMPLEX_OBSTACLES,
)


def run_profile(rrt_impl, name: str, params: Params):
    logger = NullLogger()
    _ = rrt_impl(params, logger)


if __name__ == "__main__":
    # Run profiling
    with cProfile.Profile() as pr:
        run_profile(rrt_pure_python, "rrt_pure_python", params)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    # Save results in readable format
    stats.dump_stats(filename="reports/profile.prof")  # Binary format for other tools
    with open("reports/profile_stats.txt", "w") as f:  # Text format for direct reading
        stats.stream = f
        stats.print_stats()
