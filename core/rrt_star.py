#! /usr/bin/env python3

from __future__ import annotations
from typing import Callable

import sys
from core.utils import Params
from core.logger import NullLogger, RerunLogger


def make_description(impl_type: str) -> str:
    """Returns description text for the RRT* implementation."""
    return f"""
{impl_type} implementation of RRT* algorithm in a simple environment.

You can check the reference implementation in [rerun examples](https://github.com/rerun-io/rerun/tree/main/examples/python/rrt_star).
"""


def run(
    impl_type: str,
    description: str,
    params: Params,
    rrt_fn: Callable[[Params], list | None],
) -> None:
    """Run the RRT* implementation."""
    # Use RerunLogger by default, or NullLogger if logger is None
    logger = RerunLogger()
    # logger = NullLogger()

    # Setup logger
    logger.setup(impl_type, description, params)

    # Run RRT*
    path = rrt_fn(params, logger)
    if path is not None:
        print(f"Successfully found path of size: {len(path)}")
    else:
        print("Failed to find path")

    # Teardown
    logger.teardown()


if __name__ == "__main__":
    raise NotImplementedError("This file defines the common interface for RRT* implementations. Do not run directly.")
