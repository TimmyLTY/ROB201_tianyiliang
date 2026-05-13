"""Robotics control functions."""

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Reactive obstacle avoidance.
    lidar : placebot object with lidar data
    """
    if not hasattr(reactive_obst_avoid, "is_turning"):
        reactive_obst_avoid.is_turning = False
        reactive_obst_avoid.turn_direction = 0.0
        reactive_obst_avoid.turn_timer = 0

    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    safe_distance = 20.0

    front_mask = np.abs(angles) < np.pi / 5
    front_dists = laser_dist[front_mask]
    obstacle_detected = np.any(front_dists < safe_distance)

    if obstacle_detected or reactive_obst_avoid.turn_timer > 0:
        if not reactive_obst_avoid.is_turning:
            close_mask = (laser_dist < safe_distance) & front_mask
            if np.any(close_mask):
                avg_angle = np.mean(angles[close_mask])
                if avg_angle > 0:
                    reactive_obst_avoid.turn_direction = -1.0
                else:
                    reactive_obst_avoid.turn_direction = 1.0
            else:
                reactive_obst_avoid.turn_direction = 1.0 if random.random() > 0.5 else -1.0

            reactive_obst_avoid.is_turning = True
            reactive_obst_avoid.turn_timer = 8

        if not obstacle_detected:
            reactive_obst_avoid.turn_timer -= 1

        speed = 0.0
        rotation_speed = reactive_obst_avoid.turn_direction * 0.6
    else:
        reactive_obst_avoid.is_turning = False
        reactive_obst_avoid.turn_timer = 0
        speed = 0.8
        rotation_speed = random.uniform(-0.1, 0.1)

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """

    K_goal = 10.0
    K_obs = 20.0
    d_safe = 80.0
    d_transition = 100.0
    goal_threshold = 10.0
    K_omega = 0.8
    K_v = 1.0
    phi_max = np.pi / 6

    q = np.array(current_pose[:2], dtype=float)
    q_goal = np.array(goal_pose[:2], dtype=float)
    theta = current_pose[2]

    diff = q_goal - q
    d_goal = np.linalg.norm(diff)

    if d_goal < goal_threshold:
        command = {"forward": 0, "rotation": 0}
        return command

    if d_goal < d_transition:
        grad_attractive = (K_goal / d_transition) * diff
    else:
        grad_attractive = (K_goal / d_goal) * diff

    grad_repulsive = np.array([0.0, 0.0])

    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    idx_min = np.argmin(distances)
    d_obs = distances[idx_min]
    angle_obs = angles[idx_min]

    if d_obs < d_safe:
        obs_x = q[0] + d_obs * np.cos(theta + angle_obs)
        obs_y = q[1] + d_obs * np.sin(theta + angle_obs)
        q_obs = np.array([obs_x, obs_y])

        diff_obs = q_obs - q
        grad_repulsive = (K_obs / (d_obs ** 3)) * (1.0 / d_obs - 1.0 / d_safe) * diff_obs

    grad_total = grad_attractive - grad_repulsive

    desired_angle = np.arctan2(grad_total[1], grad_total[0])

    angle_error = desired_angle - theta
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    phi_r = abs(angle_error)

    rotation_speed = np.clip(K_omega * angle_error / np.pi, -1.0, 1.0)

    grad_norm = np.linalg.norm(grad_total)
    speed_amplitude = K_v * min(1.0, grad_norm / K_goal)
    if phi_r < phi_max:
        forward_speed = speed_amplitude
    else:
        forward_speed = speed_amplitude * phi_max / phi_r

    forward_speed = float(np.clip(forward_speed, 0.0, 1.0))
    rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))

    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command
