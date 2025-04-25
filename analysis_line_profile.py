#!/usr/bin/env python3

from line_profiler import LineProfiler
from core.utils import Params, SAMPLE_OBSTACLES
from core.logger import NullLogger

from core.python_impl.rrt_star_pure import rrt as rrt_pure_python
from core.python_impl.rrt_star_numpy_opt import rrt as rrt_numpy_opt


params = Params(
    max_iterations=5000,
    max_step_size=0.1,
    start_point=(0.2, 0.5),
    end_point=(1.8, 0.5),
    wall_segments=SAMPLE_OBSTACLES,
)


def run_profile(rrt_impl, name: str, params: Params):
    logger = NullLogger()
    _ = rrt_impl(params, logger)


if __name__ == "__main__":
    # Create line profiler
    profiler = LineProfiler()

    # Add the function to profile
    profiler.add_function(rrt_pure_python)
    # profiler.add_function(rrt_numpy_opt)

    # Run profiling
    profiler.runctx(
        'run_profile(rrt_pure_python, "rrt_pure_python", params)',
        # 'run_profile(rrt_numpy_opt, "rrt_numpy_opt", params)',
        globals(),
        locals()
    )

    # Save binary format for other tools
    profiler.dump_stats('reports/line_profile.lprof')

    # Save human-readable text format
    with open('reports/line_profile.txt', 'w') as f:
        profiler.print_stats(stream=f)

    # Print to console
    profiler.print_stats()