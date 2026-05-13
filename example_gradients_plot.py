"""Potential-field gradient display."""

import random

import numpy as np
from matplotlib import pyplot as plt

goal_pos = [random.randint(1, 500), random.randint(1, 300)]

# Display a grid with a static point goal and randomly placed obstacles.
x = np.arange(0, 500, 5)
y = np.arange(0, 300, 5)

n_obstacles = 5
obstacles_pos = np.zeros((n_obstacles, 2))
for k in range(n_obstacles):
    obstacles_pos[k] = [random.randint(30, 470), random.randint(30, 270)]

X, Y = np.meshgrid(x, y, indexing='ij')

grad_x = (np.zeros_like(X)).astype('float64')
grad_y = (np.zeros_like(Y)).astype('float64')

# Goal potential parameters
d_lim = 50  # Near goal radius to be influenced with a quadratic potential instead of linear
min_r = 10  # Near goal radius to be considered as reached
K_goal = 1  # Attractive coefficient (= set max attractive gradient)
# Obstacle potential parameters
d_safe = 150  # Obstacle radius beyond which the agent is no longer repelled
K_obs = 10000  # Repulsive coefficient
for i in range(len(x)):
    for j in range(len(y)):
        d_goal = np.sqrt((goal_pos[0] - X[i][j]) ** 2 + (goal_pos[1] - Y[i][j]) ** 2)

        if d_goal < min_r:
            grad_x[i][j] = 0
            grad_y[i][j] = 0
        elif d_goal < d_lim:
            grad_x[i][j] = (K_goal / d_lim) * (goal_pos[0] - X[i][j])
            grad_y[i][j] = (K_goal / d_lim) * (goal_pos[1] - Y[i][j])
        elif d_goal > d_lim:
            grad_x[i][j] = (K_goal / d_goal) * (goal_pos[0] - X[i][j])
            grad_y[i][j] = (K_goal / d_goal) * (goal_pos[1] - Y[i][j])

        for k in range(n_obstacles):
            d_obstacle = np.sqrt((obstacles_pos[k][0] - X[i][j]) ** 2 + (obstacles_pos[k][1] - Y[i][j]) ** 2)
            if d_obstacle > d_safe:
                grad_x[i][j] += 0
                grad_y[i][j] += 0
            elif d_obstacle == 0:
                grad_x[i][j] = 0
                grad_y[i][j] = 0
            else:
                grad_x[i][j] -= (K_obs / d_obstacle ** 3) * (1 / d_obstacle - 1 / d_safe) * (
                        obstacles_pos[k][0] - X[i][j])
                grad_y[i][j] -= (K_obs / d_obstacle ** 3) * (1 / d_obstacle - 1 / d_safe) * (
                        obstacles_pos[k][1] - Y[i][j])

        norm = np.sqrt(grad_x[i][j] ** 2 + grad_y[i][j] ** 2)
        if norm > 1.0:
            grad_x[i][j] = grad_x[i][j] / norm
            grad_y[i][j] = grad_y[i][j] / norm

fig, ax = plt.subplots(figsize=(len(x), len(y)))
ax.quiver(X, Y, grad_x, grad_y)
ax.add_patch(plt.Circle(goal_pos, min_r, color='y'))
ax.annotate("Goal", xy=goal_pos, fontsize=10, ha="center")

for k in range(n_obstacles):
    ax.add_patch(plt.Circle(obstacles_pos[k], min_r, color='m'))
    ax.annotate("Obstacle", xy=obstacles_pos[k], fontsize=8, ha="center")

ax.set_title('Combined Potential when Goal and Obstacle are different ')

plt.show()
