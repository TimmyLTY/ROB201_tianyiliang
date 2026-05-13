"""
Planner class
Implementation of A*
"""

import heapq
import math
from collections import deque

import cv2
import numpy as np

try:
    from place_bot.simulation.utils.constants import ROBOT_DEFAULT_RADIUS
    from place_bot.simulation.utils.utils import circular_kernel
except ModuleNotFoundError:
    ROBOT_DEFAULT_RADIUS = 10

    def circular_kernel(radius: int) -> np.ndarray:
        """Create a filled circular kernel when Place-Bot helpers are unavailable."""
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel[x ** 2 + y ** 2 <= radius ** 2] = 1
        return kernel

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None
        self.obstacle_threshold = 0.3
        self.robot_radius_world = ROBOT_DEFAULT_RADIUS
        self.wall_dilation_margin = 3.0

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _map(self):
        """Return the planning map, or the live occupancy map if no copy exists."""
        if self.map_walls is None:
            return self.grid.occupancy_map
        return self.map_walls

    def _in_bounds(self, cell):
        """Check whether a grid cell is inside the map."""
        x, y = cell
        return 0 <= x < self.grid.x_max_map and 0 <= y < self.grid.y_max_map

    def _is_free(self, cell):
        """Check whether a grid cell is traversable for planning."""
        if not self._in_bounds(cell):
            return False
        x, y = cell
        return self._map()[x, y] <= self.obstacle_threshold

    def _nearest_free_cell(self, cell, max_radius=40):
        """Find a nearby free cell if start or goal lies on an occupied cell."""
        if self._is_free(cell):
            return cell

        queue = deque([(cell, 0)])
        visited = {cell}
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

        while queue:
            current, dist = queue.popleft()
            if dist >= max_radius:
                continue

            for dx, dy in offsets:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in visited or not self._in_bounds(neighbor):
                    continue
                if self._is_free(neighbor):
                    return neighbor
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

        return None

    def get_neighbors(self, current_cell):
        """
        Return the 8 free neighbor cells around current_cell.
        current_cell : tuple (x, y) in map/grid coordinates
        """
        neighbors = []
        x, y = current_cell
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

        for dx, dy in offsets:
            neighbor = (x + dx, y + dy)
            if self._is_free(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def heuristic(self, cell_1, cell_2):
        """Return the Euclidean distance between two map cells."""
        return math.hypot(cell_1[0] - cell_2[0], cell_1[1] - cell_2[1])

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from A* parents and convert it to world coordinates."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)

        path_map = np.array(total_path)
        path_world_x, path_world_y = self.grid.conv_map_to_world(path_map[:, 0], path_map[:, 1])
        path_world_theta = np.zeros_like(path_world_x)
        return np.vstack((path_world_x, path_world_y, path_world_theta))

    def _dilate_walls(self):
        """
        Inflate occupied cells in self.map_walls so A* keeps a safety margin from walls.
        The live SLAM map is left unchanged; only the planning copy is modified.
        """
        obstacle_mask = self.map_walls > self.obstacle_threshold

        radius_map = int(np.ceil((self.robot_radius_world + self.wall_dilation_margin)
                                 / self.grid.resolution))
        kernel = circular_kernel(max(radius_map, 1))

        # filter2D marks every cell touched by the circular robot footprint around an obstacle.
        obstacle_count = cv2.filter2D(obstacle_mask.astype(np.uint8),
                                      ddepth=cv2.CV_16U,
                                      kernel=kernel)
        dilated_mask = obstacle_count > 0
        self.map_walls[dilated_mask] = max(float(self.map_walls.max()), self.obstacle_threshold + 1.0)

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        self.map_walls = np.array(self.grid.occupancy_map, copy=True)
        self._dilate_walls()

        start_cell = self.grid.conv_world_to_map(start[0], start[1])
        goal_cell = self.grid.conv_world_to_map(goal[0], goal[1])

        start_cell = (int(start_cell[0]), int(start_cell[1]))
        goal_cell = (int(goal_cell[0]), int(goal_cell[1]))

        if not self._in_bounds(start_cell) or not self._in_bounds(goal_cell):
            print("A*: start or goal is outside the map")
            return None

        start_cell = self._nearest_free_cell(start_cell)
        goal_cell = self._nearest_free_cell(goal_cell)
        if start_cell is None or goal_cell is None:
            print("A*: no free start or goal cell found")
            return None

        open_set = []
        heapq.heappush(open_set, (self.heuristic(start_cell, goal_cell), start_cell))

        came_from = {}
        g_score = {start_cell: 0.0}
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}
        closed_set = set()

        while open_set:
            current_f, current_cell = heapq.heappop(open_set)

            if current_cell in closed_set:
                continue
            if current_f > f_score.get(current_cell, math.inf):
                continue

            if current_cell == goal_cell:
                return self.reconstruct_path(came_from, current_cell)

            closed_set.add(current_cell)
            for neighbor in self.get_neighbors(current_cell):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, neighbor)
                if tentative_g_score < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current_cell
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("A*: failed to reach objective")
        return None

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
