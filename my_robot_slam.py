"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # TP2: manually designed path through the map corridors
        # Map spans roughly x: -556 to 547, y: -364 to 375
        # Robot starts at (439, 195) in simulator coords -> (0,0) in odometry frame
        # Waypoints are defined relative to the odometry frame (start = origin)
        self.goals = [
            np.array([0,   120, 0]),   # W1: Up toward top-right
            np.array([-250, 120, 0]),  # W2: Left toward upper-middle
            np.array([-250, -250, 0]), # W3: Down along the left of center block
            np.array([0,   -250, 0]),  # W4: Right toward bottom-right
            np.array([50,  -150, 0]),  # W5: Lower right hall
            np.array([0,   0,    0]),  # W6: Home
        ]
        self.current_goal_index = 0
        self.goal_reached_threshold = 20.0
        self.display_period = 10
        self.bootstrap_steps = 100
        self.localisation_score_threshold = 70.0
        self.last_localisation_score = 0.0
        self.debug_score_mode = False
        self.debug_score_period = 200
        self.home_goal = np.array([0.0, 0.0, 0.0])
        self.tp5_exploration_steps = 4000
        self.tp5_path = None
        self.tp5_path_index = 0
        self.tp5_path_follow_threshold = 20.0
        self.tp5_return_threshold = 25.0
        self.tp5_replan_period = 100

        self.base.linear_ratio *= 0.4
        self.base.angular_ratio *= 0.6


    def control(self):
        """
        Main control function executed at each time step
        """
        # Switch between TP1, TP2, TP3, TP4 and TP5 here
        return self.control_tp5()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Potential field navigation with multi-goal support
        """
        pose = self.odometer_values()

        # Multi-goal logic: check if current goal is reached
        if self.current_goal_index < len(self.goals):
            goal = self.goals[self.current_goal_index]
            d_to_goal = np.linalg.norm(pose[:2] - goal[:2])

            if d_to_goal < self.goal_reached_threshold:
                # Move to next goal
                self.current_goal_index += 1
                if self.current_goal_index < len(self.goals):
                    goal = self.goals[self.current_goal_index]
                else:
                    # All goals reached, stop
                    return {"forward": 0, "rotation": 0}
        else:
            # All goals reached
            return {"forward": 0, "rotation": 0}

        # Compute new command speed using potential field control
        command = potential_field_control(self.lidar(), pose, goal)

        if self.counter % self.display_period == 0:
            self.occupancy_grid.display_cv(pose, goal=goal,
                                           goals=self.goals,
                                           active_goal_index=self.current_goal_index)
        self.counter += 1

        return command

    def control_tp3(self):
        """
        Control function for TP3
        Build an occupancy map from odometry while the robot explores the map
        """
        pose = self.odometer_values()

        if pose is None:
            return {"forward": 0, "rotation": 0}

        self.corrected_pose = np.array(pose, dtype=float)
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        if self.counter % self.display_period == 0:
            self.occupancy_grid.display_cv(self.corrected_pose)

        self.counter += 1
        return reactive_obst_avoid(self.lidar())

    def control_tp4(self):
        """
        Control function for TP4
        Localise the robot against the current map before updating it
        """
        raw_pose = self.odometer_values()

        if raw_pose is None:
            return {"forward": 0, "rotation": 0}

        raw_pose = np.array(raw_pose, dtype=float)
        should_update_map = self.counter < self.bootstrap_steps

        if should_update_map:
            self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_pose)
            self.last_localisation_score = 0.0
        else:
            self.last_localisation_score = self.tiny_slam.localise(self.lidar(), raw_pose)
            self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_pose)
            should_update_map = self.last_localisation_score > self.localisation_score_threshold

        if should_update_map:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        if self.counter % self.display_period == 0:
            self.occupancy_grid.display_cv(self.corrected_pose)

        if self.debug_score_mode and self.counter % self.debug_score_period == 0:
            self.tiny_slam.debug_score_offsets(self.lidar(), raw_pose)

        self.counter += 1
        if self.debug_score_mode:
            return {"forward": 0.0, "rotation": 0.0}
        return reactive_obst_avoid(self.lidar())

    def _update_slam_estimate(self):
        """
        Update corrected pose and occupancy grid with the TP4 SLAM loop.
        Returns False when the odometer is unavailable.
        """
        raw_pose = self.odometer_values()

        if raw_pose is None:
            return False

        raw_pose = np.array(raw_pose, dtype=float)
        should_update_map = self.counter < self.bootstrap_steps

        if should_update_map:
            self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_pose)
            self.last_localisation_score = 0.0
        else:
            self.last_localisation_score = self.tiny_slam.localise(self.lidar(), raw_pose)
            self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_pose)
            should_update_map = self.last_localisation_score > self.localisation_score_threshold

        if should_update_map:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        return True

    def _follow_tp5_path(self):
        """Follow the planned path with a local potential-field controller."""
        if self.tp5_path is None:
            return reactive_obst_avoid(self.lidar())

        while self.tp5_path_index < self.tp5_path.shape[1] - 1:
            target = self.tp5_path[:, self.tp5_path_index]
            distance_to_target = np.linalg.norm(self.corrected_pose[:2] - target[:2])
            if distance_to_target >= self.tp5_path_follow_threshold:
                break
            self.tp5_path_index += 1

        target = self.tp5_path[:, self.tp5_path_index]
        return potential_field_control(self.lidar(), self.corrected_pose, target)

    def control_tp5(self):
        """
        Control function for TP5
        Explore and build a map, plan a shortest path home, then follow it
        """
        if not self._update_slam_estimate():
            return {"forward": 0, "rotation": 0}

        if self.counter < self.tp5_exploration_steps:
            if self.counter % self.display_period == 0:
                self.occupancy_grid.display_cv(self.corrected_pose, goal=self.home_goal)
            self.counter += 1
            return reactive_obst_avoid(self.lidar())

        if self.tp5_path is None and (
                self.counter == self.tp5_exploration_steps
                or (self.counter - self.tp5_exploration_steps) % self.tp5_replan_period == 0):
            self.tp5_path = self.planner.plan(self.corrected_pose, self.home_goal)
            self.tp5_path_index = 0
            if self.tp5_path is None:
                print("TP5: no path found yet, continuing exploration before replanning")

        if self.tp5_path is None:
            if self.counter % self.display_period == 0:
                self.occupancy_grid.display_cv(self.corrected_pose, goal=self.home_goal)
            self.counter += 1
            return reactive_obst_avoid(self.lidar())

        distance_to_home = np.linalg.norm(self.corrected_pose[:2] - self.home_goal[:2])
        if distance_to_home < self.tp5_return_threshold:
            self.occupancy_grid.display_cv(self.corrected_pose,
                                           goal=self.home_goal,
                                           traj=self.tp5_path[:2, :])
            return {"forward": 0.0, "rotation": 0.0}

        command = self._follow_tp5_path()
        if self.counter % self.display_period == 0:
            self.occupancy_grid.display_cv(self.corrected_pose,
                                           goal=self.home_goal,
                                           traj=self.tp5_path[:2, :])

        self.counter += 1
        return command
