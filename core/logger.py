from abc import ABC, abstractmethod
from typing import Optional
import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from core.utils import Params


class Logger(ABC):
    """Base logger class that defines the interface for all loggers"""

    @abstractmethod
    def setup(self, impl_type: str, description: str, params: Params, padding: float = 0.0) -> None:
        """Setup the logger"""
        pass

    @abstractmethod
    def log_obstacles(self, obstacles: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
        """Log the obstacles"""
        pass

    @abstractmethod
    def log_tree(self, tree, random_point, closest_node, new_point, intersects_obs):
        """Log the tree"""
        pass

    @abstractmethod
    def log_rewiring(self, close_nodes, new_point, neighborhood_size):
        """Log the rewiring"""
        pass

    @abstractmethod
    def log_path(self, segments):
        """Log the path"""
        pass

    @abstractmethod
    def set_time_sequence(self, name: str, value: int) -> None:
        """Set the time sequence value"""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup the logger"""
        pass


class NullLogger(Logger):
    """A logger that does nothing - used for profiling"""

    def setup(self, impl_type: str, description: str, params: Params, padding: float = 0.0) -> None:
        pass

    def log_obstacles(self, obstacles: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
        pass

    def log_tree(self, tree, random_point, closest_node, new_point, intersects_obs):
        pass

    def log_rewiring(self, close_nodes, new_point, neighborhood_size):
        pass

    def log_path(self, segments):
        pass

    def set_time_sequence(self, name: str, value: int) -> None:
        pass

    def teardown(self) -> None:
        pass


class RerunLogger(Logger):
    """Logger that uses Rerun for visualization"""

    def __init__(self):
        self.args: Optional[argparse.Namespace] = None

    def setup(self, impl_type: str, description: str, params: Params, padding: float = 0.0) -> None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Visualization of the path finding algorithm RRT*.")
        rr.script_add_args(parser)
        self.args = parser.parse_args()

        # Calculate visual bounds from wall segments with padding
        x_coords = [p[0] for segment in params.wall_segments for p in segment]
        y_coords = [p[1] for segment in params.wall_segments for p in segment]
        visual_bounds = [[min(x_coords) - padding, max(x_coords) + padding], [min(y_coords) - padding, max(y_coords) + padding]]

        # ReRun Layout
        blueprint = rrb.Vertical(
            rrb.TextDocumentView(name="Description", origin="/description"),
            rrb.Spatial2DView(
                name="Map",
                origin="/map",
                background=[32, 0, 16],
                visual_bounds=rrb.VisualBounds2D(x_range=visual_bounds[0], y_range=visual_bounds[1]),
            ),
            row_shares=[1, 4],
        )
        rr.script_setup(self.args, f"rrt_star_{impl_type.lower().replace(' ', '_')}", default_blueprint=blueprint)
        rr.set_time_sequence("iteration", 0)
        rr.log("description", rr.TextDocument(description, media_type=rr.MediaType.MARKDOWN), static=True)

        # Log start and end points
        rr.log("map/start", rr.Points2D([params.start_point], radii=0.02, colors=[[255, 255, 255, 255]]))
        rr.log("map/destination", rr.Points2D([params.end_point], radii=0.02, colors=[[255, 255, 0, 255]]))

    def log_obstacles(self, obstacles: list[tuple[tuple[float, float], tuple[float, float]]]) -> None:
        """Log the obstacles"""
        rr.log("map/obstacles", rr.LineStrips2D(obstacles))

    def log_tree(self, tree, random_point, closest_node, new_point, intersects_obs):
        rr.log("map/new/close_nodes", rr.Clear(recursive=False))
        rr.log(
            "map/tree/edges",
            rr.LineStrips2D(tree.segments(), radii=0.0005, colors=[0, 0, 255, 128]),
        )
        rr.log(
            "map/tree/vertices",
            rr.Points2D([node.pos for node in tree], radii=0.002),
            # So that we can see the cost at a node by hovering over it.
            rr.AnyValues(cost=np.asarray([float(node.cost) for node in tree])),
        )
        rr.log("map/new/random_point", rr.Points2D([random_point], radii=0.008))
        rr.log("map/new/closest_node", rr.Points2D([closest_node.pos], radii=0.008))
        rr.log("map/new/new_point", rr.Points2D([new_point], radii=0.008))

        color = np.array([0, 255, 0, 255]).astype(np.uint8)
        if intersects_obs:
            color = np.array([255, 0, 0, 255]).astype(np.uint8)
        rr.log(
            "map/new/new_edge",
            rr.LineStrips2D([(closest_node.pos, new_point)], colors=[color], radii=0.001),
        )

    def log_rewiring(self, close_nodes, new_point, neighborhood_size):
        rr.log("map/new/close_nodes", rr.Points2D([node.pos for node in close_nodes]))
        # Add circle visualization for neighborhood
        # Create points along the circumference of the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = np.column_stack(
            [new_point[0] + neighborhood_size * np.cos(theta), new_point[1] + neighborhood_size * np.sin(theta)]
        )
        # Close the circle by adding the first point at the end
        circle_points = np.vstack([circle_points, circle_points[0]])
        rr.log(
            "map/new/neighborhood_circle",
            rr.LineStrips2D([circle_points], colors=[[255, 255, 255, 32]], radii=[0.001]),  # Thin line
        )

    def log_path(self, segments):
        rr.log("map/path", rr.LineStrips2D(segments, radii=0.002, colors=[0, 255, 255, 255]))

    def set_time_sequence(self, name: str, value: int) -> None:
        rr.set_time_sequence(name, value)

    def teardown(self) -> None:
        if self.args is not None:
            rr.script_teardown(self.args)
