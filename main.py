import copy
import os
import random

import numpy as np

from model.measurement import MeasurementModel
from model.motion import MotionModel
from robot import Robot
from scene import Scene
from utils import absolute2relative, relative2absolute
from visualize import draw


def init_particles(number_of_particles: int, init_grid: np.ndarray) -> list[Robot]:
    particles = [None] * number_of_particles
    for i in range(number_of_particles):
        particles[i] = Robot(x, y, theta, copy.deepcopy(init_grid))
    return particles


if __name__ == "__main__":
    # Create results folder.
    output_path = "results/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Load the scene.
    scene = Scene()
    x, y, theta = scene.get_initial_state()

    # Create the robot.
    robot = Robot(x, y, theta, scene.map, sense_noise=3.0)
    prev_odo = curr_odo = robot.get_states()

    # Initialize particles based on the robot's variables.
    NUMBER_OF_PARTICLES = 100
    J_inv = 1 / NUMBER_OF_PARTICLES
    rand = random.random() * J_inv
    init_grid = np.ones((scene.height, scene.width)) * robot.prior_prob
    particles = init_particles(NUMBER_OF_PARTICLES, init_grid)

    # Create the motion model.
    motion_model = MotionModel()

    # Create the measurement model.
    measurement_model = MeasurementModel(robot.radar_range)

    # FastSLAM 1.0
    for idx, (distance, angle) in enumerate(scene.movements):
        curr_odo = robot.calculate_new_pos(distance, angle)

        z_star, free_grid_star, occupy_grid_star = robot.sense()
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        weights = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):
            # Simulate a robot motion for each particle.
            prev_pose = particles[i].get_states()
            x, y, theta = motion_model.sample(prev_odo, curr_odo, prev_pose)
            particles[i].set_states(x, y, theta)

            # Calculate the particle's weights depending on the robot's measurement.
            z, _, _ = particles[i].sense()
            weights[i] = measurement_model.get_prob(z_star, z)

            # Update the occupancy grid based on the true measurements.
            curr_pose = particles[i].get_states()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(
                np.int32
            )
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(
                np.int32
            )
            particles[i].update_occupancy_grid(free_grid, occupy_grid)

        # Normalize the weights of the particles and sort them.
        weights = weights / np.sum(weights)
        best_id = np.argsort(weights)[-1]

        # Select the best particle for the robot estimation.
        best_particle = copy.deepcopy(particles[best_id])

        # Resample the particles with a sample probability proportional to the importance weight.
        # Use low variance sampling method.
        new_particles = [None] * NUMBER_OF_PARTICLES
        c = weights[0]

        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = rand + j * J_inv
            while U > c:
                i += 1
                c += weights[i]
            new_particles[j] = copy.deepcopy(particles[i])

        # Move new particles and current states to the new iteration.
        particles = new_particles
        prev_odo = curr_odo

        # Draw the robot's path of the current iteration.
        draw(robot, idx, free_grid_star, particles, best_particle)
