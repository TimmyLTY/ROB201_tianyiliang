""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    if not hasattr(reactive_obst_avoid, "is_turning"):
        reactive_obst_avoid.is_turning = False
        reactive_obst_avoid.turn_direction = 0.0
        # Add a timer to force a minimum turn duration
        reactive_obst_avoid.turn_timer = 0

    laser_dist = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    # Define a safety distance
    safe_distance = 20.0

    # Consider the front cone of the robot (angles between -pi/4 and pi/4)
    front_mask = np.abs(angles) < np.pi / 5
    front_dists = laser_dist[front_mask]
    
    # Check if there is an obstacle in front within a safety distance
    obstacle_detected = np.any(front_dists < safe_distance)

    # 1.4 Improvement: Force a minimum turn duration when an obstacle is met
    # to prevent getting stuck in tight corners (U-shape trap)
    if obstacle_detected or reactive_obst_avoid.turn_timer > 0:
        if not reactive_obst_avoid.is_turning:
            # We just encountered an obstacle. Start turning.
            close_mask = (laser_dist < safe_distance) & front_mask
            if np.any(close_mask):
                avg_angle = np.mean(angles[close_mask])
                # If obstacle is mainly on the left (positive angle), turn right (negative rotation)
                if avg_angle > 0:
                    reactive_obst_avoid.turn_direction = -1.0
                else:
                    reactive_obst_avoid.turn_direction = 1.0
            else:
                reactive_obst_avoid.turn_direction = 1.0 if random.random() > 0.5 else -1.0
            
            reactive_obst_avoid.is_turning = True
            # Set timer for minimum turn step counts (e.g. 15 iterations)
            reactive_obst_avoid.turn_timer = 8

        if not obstacle_detected:
            # We are turning but just cleared the obstacle, countdown the timer
            reactive_obst_avoid.turn_timer -= 1

        speed = 0.0  # Stop moving forward to avoid collision
        rotation_speed = reactive_obst_avoid.turn_direction * 0.6
    else:
        # No obstacle in front and timer is depleted
        reactive_obst_avoid.is_turning = False
        reactive_obst_avoid.turn_timer = 0
        speed = 0.8  # Move forward
        # Small random rotation to explore better
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

    # ---- Parameters ----
    K_goal = 10.0         # Small starting value
    K_obs = 20.0        # Small starting value (repulsion)
    d_safe = 80.0        # Reduced safety range
    d_transition = 100.0 # Transition radius
    goal_threshold = 10.0  # Stop threshold
    K_omega = 0.8        # Rotation gain
    K_v = 1.0            # Forward gain
    phi_max = np.pi / 6  # Alignment threshold


    # ---- Current state ----
    q = np.array(current_pose[:2], dtype=float)
    q_goal = np.array(goal_pose[:2], dtype=float)
    theta = current_pose[2]

    # ---- Distance to goal ----
    diff = q_goal - q
    d_goal = np.linalg.norm(diff)

    # ---- Goal reached: stop ----
    if d_goal < goal_threshold:
        command = {"forward": 0, "rotation": 0}
        return command

    # ---- Attractive gradient ----
    # Quadratic zone (close to goal): gradient proportional to distance
    # Linear zone (far from goal): gradient with constant norm K_goal
    if d_goal < d_transition:
        # Quadratic: grad = K_goal / d_transition * (q_goal - q)
        grad_attractive = (K_goal / d_transition) * diff
    else:
        # Linear: grad = K_goal / d * (q_goal - q)
        grad_attractive = (K_goal / d_goal) * diff

    # ---- Repulsive gradient from nearest obstacle ----
    grad_repulsive = np.array([0.0, 0.0])

    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # Find the nearest obstacle
    idx_min = np.argmin(distances)
    d_obs = distances[idx_min]
    angle_obs = angles[idx_min]

    if d_obs < d_safe:
        # Position of the nearest obstacle in the absolute frame
        obs_x = q[0] + d_obs * np.cos(theta + angle_obs)
        obs_y = q[1] + d_obs * np.sin(theta + angle_obs)
        q_obs = np.array([obs_x, obs_y])

        # Repulsive gradient formula from the PDF:
        # grad_f = K_obs / d^3 * (1/d - 1/d_safe) * (q_obs - q)
        # The gradient points TOWARDS the obstacle, so we SUBTRACT it
        diff_obs = q_obs - q
        grad_repulsive = (K_obs / (d_obs ** 3)) * (1.0 / d_obs - 1.0 / d_safe) * diff_obs

    # ---- Total gradient ----
    # Attractive pulls towards goal, repulsive pushes away from obstacle
    # Since repulsive grad points towards obstacle, we subtract it
    grad_total = grad_attractive - grad_repulsive

    # ---- Convert gradient to robot commands ----
    # Desired heading from the gradient
    desired_angle = np.arctan2(grad_total[1], grad_total[0])

    # Angular error (difference between desired heading and current orientation)
    angle_error = desired_angle - theta
    # Normalize angle_error to [-pi, pi]
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    phi_r = abs(angle_error)

    # Rotation speed: proportional to angle error (slide formula: ω = K_ω * φ_r)
    rotation_speed = np.clip(K_omega * angle_error / np.pi, -1.0, 1.0)

    # Forward speed: piecewise linear attenuation (slide formula)
    # V = K_v * |F| if φ_r < φ_max
    # V = K_v * |F| * φ_max / φ_r if φ_r >= φ_max
    grad_norm = np.linalg.norm(grad_total)
    speed_amplitude = K_v * min(1.0, grad_norm / K_goal)  # normalize to [0,1]
    if phi_r < phi_max:
        forward_speed = speed_amplitude
    else:
        forward_speed = speed_amplitude * phi_max / phi_r

    # Clamp to [-1, 1]
    forward_speed = float(np.clip(forward_speed, 0.0, 1.0))
    rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))

    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command

