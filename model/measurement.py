import json

import numpy as np

from utils import normal_distribution


class MeasurementModel(object):
    def __init__(self, radar_range):
        with open("measurement.json") as f:
            data = json.load(f)

        self.p_hit = data["p_hit"]
        self.sigma_hit = data["sigma_hit"]
        self.p_short = data["p_short"]
        self.p_max = data["p_max"]
        self.p_rand = data["p_rand"]
        self.lambda_short = data["lambda_short"]
        self.radar_range = radar_range

    def get_prob(self, z_star, z) -> np.ndarray:
        z_star, z = np.array(z_star), np.array(z)

        # Probability of measuring the correct range with added local measurement noise.
        prob_hit = normal_distribution(z - z_star, np.power(self.sigma_hit, 2))

        # Probability of hitting unexpected objects.
        prob_short = self.lambda_short * np.exp(-self.lambda_short * z)
        prob_short[np.greater(z, z_star)] = 0

        # Probability of not hitting anything or particle failure.
        prob_max = np.zeros_like(z)
        prob_max[z == self.radar_range] = 1

        # Probability of random measurements.
        prob_rand = 1 / self.radar_range

        # Total probability (hit + short + max + random = 1).
        prob = (
            self.p_hit * prob_hit
            + self.p_short * prob_short
            + self.p_max * prob_max
            + self.p_rand * prob_rand
        )
        prob = np.prod(prob)

        return prob
