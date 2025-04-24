#! /usr/bin/env python3

from __future__ import annotations
from typing import Generator, Tuple, List

import math
import random
import numpy as np
from numba import njit, typed, types

from core.rrt_star import make_description, run
from core.utils import SAMPLE_OBSTACLES, Params
from core.logger import Logger

IMPL_TYPE = "Numba (Numpy)"
DESCRIPTION = make_description(IMPL_TYPE)

Point2D = Tuple[float, float]
Line2D = Tuple[Point2D, Point2D]


@njit
def distance(point0: np.ndarray, point1: np.ndarray) -> float:
    return math.sqrt((point0[0] - point1[0]) ** 2 + (point0[1] - point1[1]) ** 2)


@njit
def steer(start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    """Finds the point in a disc around `start` that is closest to `end`."""
    dist = distance(start, end)
    if dist < radius:
        return end
    else:
        diff = end - start
        norm = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        direction = diff / norm
        return direction * radius + start


@njit
def segments_intersect(start0: np.ndarray, end0: np.ndarray, start1: np.ndarray, end1: np.ndarray) -> bool:
    """Checks if the segments (start0, end0) and (start1, end1) intersect."""
    dir0 = end0 - start0
    dir1 = end1 - start1
    det = (dir0[0] * dir1[1]) - (dir0[1] * dir1[0])
    if abs(det) <= 0.00001:  # They are close to perpendicular
        return False

    # Calculate s and t parameters
    dx = start1[0] - start0[0]
    dy = start1[1] - start0[1]
    s = (dx * dir1[1] - dy * dir1[0]) / det
    t = (dx * dir0[1] - dy * dir0[0]) / det
    return (0 <= s <= 1) and (0 <= t <= 1)


class Node:
    parent: Node | None
    pos: np.ndarray
    cost: float
    children: List[Node]

    def __init__(self, parent: Node | None, position: np.ndarray, cost: float) -> None:
        self.parent = parent
        self.pos = position
        self.cost = cost
        self.children = []

    def change_cost(self, delta_cost: float) -> None:
        """Modifies the cost of this node and all child nodes."""
        self.cost += delta_cost
        for child_node in self.children:
            child_node.change_cost(delta_cost)


class RRTTree:
    root: Node

    def __init__(self, root_pos: np.ndarray) -> None:
        self.root = Node(None, root_pos, 0)

    def __iter__(self) -> Generator[Node, None, None]:
        nxt = [self.root]
        while len(nxt) >= 1:
            cur = nxt.pop()
            yield cur
            for child in cur.children:
                nxt.append(child)

    def segments(self) -> List[Line2D]:
        """Returns all the edges of the tree."""
        strips = []
        for node in self:
            if node.parent is not None:
                start = node.pos
                end = node.parent.pos
                strips.append((tuple(start), tuple(end)))
        return strips

    def nearest(self, point: np.ndarray) -> Node:
        """Finds the point in the tree that is closest to `point`."""
        min_dist = distance(point, self.root.pos)
        closest_node = self.root
        for node in self:
            dist = distance(point, node.pos)
            if dist < min_dist:
                closest_node = node
                min_dist = dist
        return closest_node

    def add_node(self, parent: Node, node: Node) -> None:
        parent.children.append(node)
        node.parent = parent

    def in_neighborhood(self, point: np.ndarray, radius: float) -> List[Node]:
        return [node for node in self if distance(node.pos, point) < radius]


def path_to_root(node: Node) -> List[np.ndarray]:
    path = [node.pos]
    cur_node = node
    while cur_node.parent is not None:
        cur_node = cur_node.parent
        path.append(cur_node.pos)
    return path


def rrt(params: Params, logger: Logger) -> List[Line2D] | None:
    # Unpack parameters
    start = np.array(params.start_point)
    end = np.array(params.end_point)
    max_step_size: float = params.max_step_size
    neighborhood_size: float = params.neighborhood_size
    max_iter: int | None = params.max_iterations

    # Convert obstacles to numpy arrays
    obstacles = [(np.array(start), np.array(end)) for start, end in params.wall_segments]
    logger.log_obstacles(obstacles)

    # Initialize tree
    tree = RRTTree(start)

    # Main loop
    path = None
    iter = 0
    end_node = None
    iter_found = None

    while iter_found is None or iter < iter_found * 3:
        random_point = np.array([random.random() * 2, random.random()])
        closest_node = tree.nearest(random_point)
        new_point = steer(closest_node.pos, random_point, max_step_size)

        # Check for obstacle intersections
        intersects_obs = False
        for obs_start, obs_end in obstacles:
            if segments_intersect(closest_node.pos, new_point, obs_start, obs_end):
                intersects_obs = True
                break

        iter += 1
        if max_iter is not None and iter >= max_iter:
            print("Max iterations reached")
            break
        logger.set_time_sequence("iteration", iter)
        logger.log_tree(tree, tuple(random_point), closest_node, tuple(new_point), intersects_obs)

        if not intersects_obs:
            close_nodes = tree.in_neighborhood(new_point, neighborhood_size)
            logger.log_rewiring(close_nodes, tuple(new_point), neighborhood_size)

            min_node = min(
                filter(
                    lambda node: not any(
                        segments_intersect(node.pos, new_point, obs_start, obs_end) for obs_start, obs_end in obstacles
                    ),
                    close_nodes + [closest_node],
                ),
                key=lambda node: node.cost + distance(node.pos, new_point),
            )

            cost = distance(min_node.pos, new_point)
            added_node = Node(min_node, new_point, cost + min_node.cost)
            tree.add_node(min_node, added_node)

            # Rewiring
            for node in close_nodes:
                cost = added_node.cost + distance(added_node.pos, node.pos)
                if (
                    not any(segments_intersect(new_point, node.pos, obs_start, obs_end) for obs_start, obs_end in obstacles)
                    and cost < node.cost
                ):
                    parent = node.parent
                    if parent is not None:
                        parent.children.remove(node)
                        node.parent = added_node
                        node.change_cost(cost - node.cost)
                        added_node.children.append(node)

            if (
                distance(new_point, end) < max_step_size
                and not any(segments_intersect(new_point, end, obs_start, obs_end) for obs_start, obs_end in obstacles)
                and end_node is None
            ):
                end_node = Node(added_node, end, added_node.cost + distance(new_point, end))
                tree.add_node(added_node, end_node)
                iter_found = iter

            if end_node:
                path = path_to_root(end_node)
                segments = [(tuple(path[i]), tuple(path[i + 1])) for i in range(len(path) - 1)]
                logger.log_path(segments)

    return path


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
