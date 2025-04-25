# Performance RRT* for Educational Purposes

This repository contains RRT* (Rapidly-exploring Random Tree Star) implementations in Python, NumPy, Numba, C++, and SIMD-optimized C++.
These implementations are used to compare performance across different programming approaches.
The code uses cProfile and line_profiler to profile and analyze the performance of each implementation.
This code is primarily written for educational purposes and is released under the MIT License.
It can be freely used, modified, and distributed in your own projects.

Rerun visualization

```sh
uv run .\core\python_impl\rrt_star_pure.py
uv run .\core\python_impl\rrt_star_numba.py
...
uv run .\core\cpp_impl\rrt_star_cpp_full.py
uv run .\core\cpp_impl\rrt_star_simd_full.py
...
```

Benchmark

```sh
uv run .\benchmark_rrt.py --scenario="simple"
uv run .\benchmark_rrt.py --scenario="complex"
```

cProfile

```sh
uv run .\analysis_cprofile.py
uv run snakeviz reports/profile.prof
cat reports/profile_stats.txt
```

line_profiler

```sh
uv run .\analysis_line_profile.py
cat reports/line_profile.txt
```

Compile C++ bindings

```sh
# Rebuild C++
uv sync --reinstall
```

The reference implementation was based on [rerun examples](https://github.com/rerun-io/rerun/tree/main/examples/python/rrt_star).
