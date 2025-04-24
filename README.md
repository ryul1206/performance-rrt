# Performance RRT* for Educational Purposes

This repository contains RRT* (Rapidly-exploring Random Tree Star) implementations in Python, NumPy, Numba, C++, and SIMD-optimized C++.
These implementations are used to compare performance across different programming approaches.
It uses cProfile and line_profiler to profile and analyze the performance of each implementation.
This code is primarily written for educational purposes and is released under the MIT License.
It can be freely used, modified, and distributed in your own projects.

```sh
uv run .\benchmark_rrt.py --scenario="simple"
uv run .\benchmark_rrt.py --scenario="complex"
```

```sh
uv run .\analysis_cprofile.py
uv run snakeviz reports/profile.prof
cat reports/profile_stats.txt
```

```sh
uv run .\analysis_line_profile.py
cat reports/line_profile.txt
```

```sh
# Rebuild C++
uv sync --reinstall
```

The reference implementation was based on [rerun examples](https://github.com/rerun-io/rerun/tree/main/examples/python/rrt_star).
