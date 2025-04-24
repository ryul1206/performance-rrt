#! /usr/bin/env python3

from __future__ import annotations
from typing import Generator, Tuple

import random
from core.cpp_impl.rrt_star_cpp_fn_bind import distance, segments_intersect, steer

from core.rrt_star import make_description, run
from core.utils import SAMPLE_OBSTACLES, Params
from core.logger import Logger


IMPL_TYPE = "C++ Bindings"
DESCRIPTION = make_description(IMPL_TYPE)


Point2D = Tuple[float, float]
Line2D = Tuple[Point2D, Point2D]


# def distance(point0: Point2D, point1: Point2D) -> float:
#     return cpp_distance(point0, point1)


# def steer(start: Point2D, end: Point2D, radius: float) -> Point2D:
#     return cpp_steer(start, end, radius)


class Map:
    obstacles: list[Line2D]

    def __init__(self, obstacles: list[Line2D] = None) -> None:
        self.obstacles = []  # List of lines as tuples of  (start, end)
        for start, end in obstacles:
            self.obstacles.append((start, end))

    def intersects_obstacle(self, start: Point2D, end: Point2D) -> bool:
        return not all(not segments_intersect(start, end, obs_start, obs_end) for (obs_start, obs_end) in self.obstacles)

    # def segments_intersect(self, start0: Point2D, end0: Point2D, start1: Point2D, end1: Point2D) -> bool:
    #     return cpp_segments_intersect(start0, end0, start1, end1)


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

    def __init__(self, root_pos: Point2D) -> None:
        self.root = Node(None, root_pos, 0)

    def __iter__(self) -> Generator[Node, None, None]:  # THIS(prime)
        nxt = [self.root]
        while len(nxt) >= 1:
            cur = nxt.pop()
            yield cur
            for child in cur.children:
                nxt.append(child)

    def segments(self) -> list[Line2D]:
        """Returns all the edges of the tree."""
        strips = []
        for node in self:
            if node.parent is not None:
                start = node.pos
                end = node.parent.pos
                strips.append((start, end))
        return strips

    def nearest(self, point: Point2D) -> Node:  # THIS(backlog)
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

    def in_neighborhood(self, point: Point2D, radius: float) -> list[Node]:
        return [node for node in self if distance(node.pos, point) < radius]


def path_to_root(node: Node) -> list[Point2D]:
    path = [node.pos]
    cur_node = node
    while cur_node.parent is not None:
        cur_node = cur_node.parent
        path.append(cur_node.pos)
    return path


def rrt(params: Params, logger: Logger) -> list[Line2D] | None:
    # Unpack parameters
    start: Point2D = params.start_point
    end: Point2D = params.end_point
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
        random_point = (random.random() * 2, random.random())
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
        # wall_segments=COMPLEX_OBSTACLES,
    )
    run(IMPL_TYPE, DESCRIPTION, params, rrt)


if __name__ == "__main__":
    main()
