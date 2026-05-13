""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        self.localisation_sigma = np.array([6.0, 6.0, np.deg2rad(3.0)])
        self.localisation_max_no_improve = 30

    @staticmethod
    def _normalize_angle(angle):
        """Wrap an angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _compose_pose(ref_pose, local_pose):
        """Compose a pose expressed in the odom frame with the odom frame pose in the map."""
        cos_ref = np.cos(ref_pose[2])
        sin_ref = np.sin(ref_pose[2])

        x_world = ref_pose[0] + cos_ref * local_pose[0] - sin_ref * local_pose[1]
        y_world = ref_pose[1] + sin_ref * local_pose[0] + cos_ref * local_pose[1]
        theta_world = TinySlam._normalize_angle(ref_pose[2] + local_pose[2])

        return np.array([x_world, y_world, theta_world])

    @staticmethod
    def _extract_valid_lidar_data(lidar):
        """Return lidar ranges and angles corresponding to obstacle hits only."""
        ranges = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()

        if ranges is None or ray_angles is None:
            empty = np.array([], dtype=float)
            return empty, empty

        max_range = float(lidar.max_range)
        valid = np.isfinite(ranges)
        valid &= ranges > 0.0
        valid &= ranges < (max_range - 1.0)

        return ranges[valid], ray_angles[valid]

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        ranges, ray_angles = self._extract_valid_lidar_data(lidar)

        if ranges.size == 0:
            return 0.0

        world_angles = pose[2] + ray_angles
        hit_points_x = pose[0] + ranges * np.cos(world_angles)
        hit_points_y = pose[1] + ranges * np.sin(world_angles)

        map_x, map_y = self.grid.conv_world_to_map(hit_points_x, hit_points_y)
        valid = np.logical_and(np.logical_and(map_x >= 0, map_x < self.grid.x_max_map),
                               np.logical_and(map_y >= 0, map_y < self.grid.y_max_map))

        if not np.any(valid):
            return 0.0

        score = np.sum(self.grid.occupancy_map[map_x[valid], map_y[valid]])
        return float(score)

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        return self._compose_pose(np.asarray(odom_pose_ref, dtype=float),
                                  np.asarray(odom_pose, dtype=float))

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        best_ref = np.array(self.odom_pose_ref, dtype=float)
        best_pose = self.get_corrected_pose(raw_odom_pose, best_ref)
        best_score = self._score(lidar, best_pose)

        no_improve = 0
        while no_improve < self.localisation_max_no_improve:
            candidate_ref = best_ref + np.random.normal(0.0, self.localisation_sigma)
            candidate_ref[2] = self._normalize_angle(candidate_ref[2])

            candidate_pose = self.get_corrected_pose(raw_odom_pose, candidate_ref)
            candidate_score = self._score(lidar, candidate_pose)

            if candidate_score > best_score:
                best_score = candidate_score
                best_ref = candidate_ref
                no_improve = 0
            else:
                no_improve += 1

        self.odom_pose_ref = best_ref
        return float(best_score)

    def debug_score_offsets(self, lidar, raw_odom_pose,
                            offsets_xy=None,
                            offsets_theta_deg=None):
        """
        Print scores for systematic offsets around the current odom reference.
        Useful to assess whether the score function has a clear optimum.
        """
        if offsets_xy is None:
            offsets_xy = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
        if offsets_theta_deg is None:
            offsets_theta_deg = [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0]

        ref = np.array(self.odom_pose_ref, dtype=float)
        raw_odom_pose = np.asarray(raw_odom_pose, dtype=float)

        print("\n===== SCORE DEBUG =====")
        print("odom_pose_ref:", np.round(ref, 3))

        base_pose = self.get_corrected_pose(raw_odom_pose, ref)
        base_score = self._score(lidar, base_pose)
        print("base_score:", round(base_score, 3))

        xy_scores = np.zeros((len(offsets_xy), len(offsets_xy)))
        best_score = base_score
        best_offset = np.array([0.0, 0.0, 0.0])

        for iy, dy in enumerate(offsets_xy):
            for ix, dx in enumerate(offsets_xy):
                candidate_ref = ref + np.array([dx, dy, 0.0])
                candidate_pose = self.get_corrected_pose(raw_odom_pose, candidate_ref)
                score = self._score(lidar, candidate_pose)
                xy_scores[iy, ix] = score
                if score > best_score:
                    best_score = score
                    best_offset = np.array([dx, dy, 0.0])

        print("XY score grid (rows=dy, cols=dx):")
        header = "dy\\dx " + " ".join(f"{dx:8.1f}" for dx in offsets_xy)
        print(header)
        for dy, row in zip(offsets_xy, xy_scores):
            print(f"{dy:5.1f} " + " ".join(f"{val:8.1f}" for val in row))

        theta_scores = []
        for dtheta_deg in offsets_theta_deg:
            candidate_ref = ref + np.array([0.0, 0.0, np.deg2rad(dtheta_deg)])
            candidate_ref[2] = self._normalize_angle(candidate_ref[2])
            candidate_pose = self.get_corrected_pose(raw_odom_pose, candidate_ref)
            score = self._score(lidar, candidate_pose)
            theta_scores.append((dtheta_deg, score))
            if score > best_score:
                best_score = score
                best_offset = np.array([0.0, 0.0, dtheta_deg])

        print("Theta scores (deg -> score):")
        print(" ".join(f"{deg:+5.1f}:{score:7.1f}" for deg, score in theta_scores))

        print("best_debug_offset:", np.round(best_offset, 3),
              "best_score:", round(best_score, 3))

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        ranges, ray_angles = self._extract_valid_lidar_data(lidar)

        if ranges.size == 0:
            return

        # Simple inverse sensor model tuned for TP3 occupancy mapping.
        free_update = -0.25
        occupied_update = 0.9
        neutral_band = 10.0
        map_min = -4.0
        map_max = 4.0

        world_angles = pose[2] + ray_angles
        cos_angles = np.cos(world_angles)
        sin_angles = np.sin(world_angles)

        hit_points_x = pose[0] + ranges * cos_angles
        hit_points_y = pose[1] + ranges * sin_angles

        free_ranges = np.maximum(ranges - neutral_band, 0.0)
        free_select = free_ranges > 0.0
        free_points_x = pose[0] + free_ranges[free_select] * cos_angles[free_select]
        free_points_y = pose[1] + free_ranges[free_select] * sin_angles[free_select]

        for x_end, y_end in zip(free_points_x, free_points_y):
            self.grid.add_value_along_line(pose[0], pose[1], x_end, y_end, free_update)

        self.grid.add_map_points(hit_points_x, hit_points_y, occupied_update)
        np.clip(self.grid.occupancy_map, map_min, map_max, out=self.grid.occupancy_map)

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
