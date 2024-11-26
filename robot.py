import json

import numpy as np

from utils import bresenham, logodds2prob, normalize_rad, prob2logodds


class Robot:
    def __init__(self, x, y, theta, grid, sense_noise=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = []

        self.grid = grid
        self.grid_size = self.grid.shape

        with open("robot.json") as f:
            data = json.load(f)

            self.prior_prob = data["prior_prob"]
            self.occupy_prob = data["occupy_prob"]
            self.free_prob = data["free_prob"]
            self.num_sensors = data["num_sensors"]
            self.radar_length = data["radar_length"]
            self.radar_range = data["radar_range"]

        self.radar_theta = (
            np.arange(0, self.num_sensors) * (2 * np.pi / self.num_sensors)
            + np.pi / self.num_sensors
        )

        # Noise for robot movement.
        self.sense_noise = sense_noise if sense_noise is not None else 0.0

    def get_states(self) -> tuple[int, int, int]:
        return self.x, self.y, self.theta

    def set_states(self, x, y, theta) -> None:
        self.x = x
        self.y = y
        self.theta = theta

        self.trajectory.append([self.x, self.y])

    def calculate_new_pos(self, distance, angle) -> tuple[int, int, int]:
        current_rad = np.radians(self.theta)
        change_rad = np.radians(angle)

        new_rad = current_rad + change_rad
        normalized_rad = normalize_rad(new_rad)

        self.x = self.x + (distance * np.cos(normalized_rad))
        self.y = self.y + (distance * np.sin(normalized_rad))
        self.theta = np.rad2deg(new_rad)

        self.trajectory.append([self.x, self.y])

        return self.x, self.y, self.theta

    def update_occupancy_grid(self, free_grid, occupy_grid) -> None:
        mask1 = np.logical_and(
            0 < free_grid[:, 0], free_grid[:, 0] < self.grid.shape[1]
        )
        mask2 = np.logical_and(
            0 < free_grid[:, 1], free_grid[:, 1] < self.grid.shape[0]
        )
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = (
            prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]])
            + prob2logodds(inverse_prob)
            - prob2logodds(self.prior_prob)
        )
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(
            0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid.shape[1]
        )
        mask2 = np.logical_and(
            0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid.shape[0]
        )
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = (
            prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]])
            + prob2logodds(inverse_prob)
            - prob2logodds(self.prior_prob)
        )
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)

    def sense(self) -> tuple[list, list, list]:
        measurements, free_grid, occupy_grid = self.ray_casting()
        measurements = np.clip(
            measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors),
            0.0,
            self.radar_range,
        )

        return measurements, free_grid, occupy_grid

    def ray_casting(self):
        measurements = [self.radar_range] * self.num_sensors
        loc = np.array([self.x, self.y])

        free_grid = []
        occupy_grid = []
        beams = self.build_radar_beams()
        for i, beam in enumerate(beams):
            dist = np.linalg.norm(beam - loc, axis=1)
            beam = np.array(beam)

            obstacle_position = np.nonzero(self.grid[beam[:, 1], beam[:, 0]] >= 0.9)[0]
            if len(obstacle_position) > 0:
                idx = obstacle_position[0]
                occupy_grid.append(list(beam[idx]))
                free_grid.extend(list(beam[:idx]))
                measurements[i] = dist[idx]
            else:
                free_grid.extend(list(beam))

        return measurements, free_grid, occupy_grid

    def build_radar_beams(self) -> list[list[list[int, int]]]:
        radar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        radar_theta = self.radar_theta + self.theta
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta) * self.radar_length,
                np.sin(radar_theta) * self.radar_length,
            ),
            axis=0,
        )

        radar_dest = radar_rel_dest + radar_src

        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = radar_src[:, i]
            x2, y2 = radar_dest[:, i]
            beams[i] = bresenham(x1, y1, x2, y2, self.grid.shape[0], self.grid.shape[1])

        return beams

    def inverse_sensing_model(self, occupy) -> float:
        if occupy:
            return self.occupy_prob
        return self.free_prob
