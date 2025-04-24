#! /usr/bin/env python3

from __future__ import annotations
from typing import Generator, Tuple, Annotated, Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from core.rrt_star import make_description, run
from core.utils import SAMPLE_OBSTACLES, Params
from core.logger import Logger

IMPL_TYPE = "Numba Numpy Optimized"
DESCRIPTION = make_description(IMPL_TYPE)

Point2D = Annotated[npt.NDArray[np.float64], Literal[2]]
Line2D = Tuple[Point2D, Point2D]

@njit
def distance(point0: Point2D, point1: Point2D) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(point0 - point1))

@njit
def distances_batch(points: np.ndarray, target: Point2D) -> np.ndarray:
    """
    Calculate Euclidean distances between multiple points and a target point.
    """
    return np.linalg.norm(points - target, axis=1)

@njit
def segments_intersect(start0: Point2D, end0: Point2D, start1: Point2D, end1: Point2D) -> bool:
    """
    Checks if the segments (start0, end0) and (start1, end1) intersect.
    Uses reusable temporary arrays to avoid memory allocation.
    """
    # Reusable temporary arrays
    dir0 = np.zeros(2, dtype=np.float64)
    dir1 = np.zeros(2, dtype=np.float64)
    mat = np.zeros((2, 2), dtype=np.float64)

    # Calculate direction vectors
    dir0[:] = end0 - start0
    dir1[:] = end1 - start1

    # Build matrix for solving linear system
    mat[:, 0] = dir0
    mat[:, 1] = dir1

    # Check if segments are parallel
    if abs(np.linalg.det(mat)) <= 0.00001:
        return False

    # Solve for intersection parameters
    s, t = np.linalg.solve(mat, start1 - start0)
    return (0 <= float(s) <= 1) and (0 <= -float(t) <= 1)

@njit
def steer(start: Point2D, end: Point2D, radius: float) -> Point2D:
    """Finds the point in a disc around `start` that is closest to `end`."""
    diff = end - start
    dist = np.linalg.norm(diff)
    if dist < radius:
        return end
    return start + (diff / dist) * radius

class Map:
    obstacles: list[Line2D]

    def __init__(self, obstacles: list[Line2D] = None) -> None:
        self.obstacles = []  # List of lines as tuples of (start, end)
        for start, end in obstacles:
            self.obstacles.append((np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)))

    def intersects_obstacle(self, start: Point2D, end: Point2D) -> bool:
        """Check if the segment (start, end) intersects any obstacle."""
        return not all(not segments_intersect(start, end, obs_start, obs_end) for (obs_start, obs_end) in self.obstacles)

class Node:
    parent: Node | None
    pos: Point2D
    cost: float
    children: list[Node]

    def __init__(self, parent: Node | None, position: Point2D, cost: float) -> None:
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
    _all_node_positions: np.ndarray  # Cache for all node positions in the tree, shape: (max_nodes, 2)
    _nodes: list[Node]  # List of nodes in the same order as _all_node_positions
    _num_nodes: int  # Current number of nodes in the tree
    _max_nodes: int  # Maximum number of nodes that can be stored

    def __init__(self, root_pos: Point2D, max_nodes: int = 10000) -> None:
        self.root = Node(None, root_pos, 0)
        self._max_nodes = max_nodes
        # Pre-allocate arrays
        self._all_node_positions = np.zeros((max_nodes, 2), dtype=np.float64)
        self._all_node_positions[0] = root_pos
        self._nodes = [self.root]
        self._num_nodes = 1

    def __iter__(self) -> Generator[Node, None, None]:
        nxt = [self.root]
        while len(nxt) >= 1:
            cur = nxt.pop()
            yield cur
            for child in cur.children:
                nxt.append(child)

    def segments(self) -> list[Line2D]:
        """Returns all the edges of the tree."""
        strips = [(node.pos, node.parent.pos) for node in self if node.parent is not None]
        return strips

    def nearest(self, point: Point2D) -> Node:
        """Finds the point in the tree that is closest to `point`."""
        # Vectorized distance calculation using cached positions
        # Use squared distances to avoid sqrt
        diff = self._all_node_positions[:self._num_nodes] - point
        squared_distances = np.sum(diff * diff, axis=1)
        min_idx = np.argmin(squared_distances)
        return self._nodes[min_idx]

    def add_node(self, parent: Node, node: Node) -> None:
        if self._num_nodes >= self._max_nodes:
            # Double the size of the arrays if needed
            new_max = self._max_nodes * 2
            new_positions = np.zeros((new_max, 2), dtype=np.float64)
            new_positions[:self._max_nodes] = self._all_node_positions
            self._all_node_positions = new_positions
            self._max_nodes = new_max

        parent.children.append(node)
        node.parent = parent
        # Update position cache with new node
        self._all_node_positions[self._num_nodes] = node.pos
        self._nodes.append(node)
        self._num_nodes += 1

    def in_neighborhood(self, point: Point2D, radius: float) -> list[Node]:
        # Vectorized distance calculation using cached positions
        # Use squared distances to avoid sqrt
        diff = self._all_node_positions[:self._num_nodes] - point
        squared_distances = np.sum(diff * diff, axis=1)
        squared_radius = radius * radius
        indices = np.where(squared_distances < squared_radius)[0]
        return [self._nodes[idx] for idx in indices]

def path_to_root(node: Node) -> list[Point2D]:
    path = [node.pos]
    cur_node = node
    while cur_node.parent is not None:
        cur_node = cur_node.parent
        path.append(cur_node.pos)
    return path

def rrt(params: Params, logger: Logger) -> list[Line2D] | None:
    # Unpack parameters
    start: Point2D = np.array(params.start_point, dtype=np.float64)
    end: Point2D = np.array(params.end_point, dtype=np.float64)
    max_step_size: float = params.max_step_size
    neighborhood_size: float = params.neighborhood_size
    max_iter: int | None = params.max_iterations

    # Obstacles
    mp = Map(obstacles=params.wall_segments)
    logger.log_obstacles(mp.obstacles)

    # Initialize tree
    tree = RRTTree(start)

    # Main loop
    path = None
    iter = 0  # How many iterations of the algorithm we have done.
    end_node = None
    iter_found = None  # Iteration when the path was last found

    while iter_found is None or iter < iter_found * 3:
        random_point = np.random.rand(2) * np.array([2, 1])
        closest_node = tree.nearest(random_point)
        new_point = steer(closest_node.pos, random_point, max_step_size)
        intersects_obs = mp.intersects_obstacle(closest_node.pos, new_point)

        iter += 1
        if max_iter is not None and iter >= max_iter:
            print("Max iterations reached")
            break
        logger.set_time_sequence("iteration", iter)
        logger.log_tree(tree, random_point, closest_node, new_point, intersects_obs)

        if not intersects_obs:
            # Searches for the point in a neighborhood that would result in the minimal cost (distance from start).
            close_nodes = tree.in_neighborhood(new_point, neighborhood_size)
            logger.log_rewiring(close_nodes, new_point, neighborhood_size)

            min_node = min(
                filter(
                    lambda node: not mp.intersects_obstacle(node.pos, new_point),
                    close_nodes + [closest_node],
                ),
                key=lambda node: node.cost + distance(node.pos, new_point),
            )

            cost = distance(min_node.pos, new_point)
            added_node = Node(min_node, new_point, cost + min_node.cost)
            tree.add_node(min_node, added_node)

            # (Rewiring) Modifies nearby nodes that would be reached faster by going through `added_node`.
            for node in close_nodes:
                cost = added_node.cost + distance(added_node.pos, node.pos)
                if not mp.intersects_obstacle(new_point, node.pos) and cost < node.cost:
                    parent = node.parent
                    if parent is not None:
                        parent.children.remove(node)

                        node.parent = added_node
                        node.change_cost(cost - node.cost)
                        added_node.children.append(node)

            if distance(new_point, end) < max_step_size and not mp.intersects_obstacle(new_point, end) and end_node is None:
                end_node = Node(added_node, end, added_node.cost + distance(new_point, end))
                tree.add_node(added_node, end_node)
                iter_found = iter

            if end_node:
                # Reconstruct shortest path in tree
                path = path_to_root(end_node)
                segments = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
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