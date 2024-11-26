import matplotlib.pyplot as plt
import numpy as np


def normalize_rad(radian) -> float:
    return radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))


def normal_distribution(mean, variance) -> np.ndarray:
    return np.exp(
        -(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance)
    )


def create_rotation_matrix(theta) -> tuple[np.ndarray, np.ndarray]:
    mat_rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    mat_rot_inv = np.linalg.inv(mat_rot)

    return mat_rot, mat_rot_inv


def absolute2relative(position, states) -> np.ndarray:
    x, y, theta = states
    pose = np.array([x, y])

    _, mat_rot_inv = create_rotation_matrix(theta)
    position = position - pose
    position = np.array(position) @ mat_rot_inv.T

    return position


def relative2absolute(position, states) -> np.ndarray:
    x, y, theta = states
    pose = np.array([x, y])

    mat_rot, _ = create_rotation_matrix(theta)
    position = np.array(position) @ mat_rot.T
    position = position + pose

    return position


def prob2logodds(prob) -> np.ndarray:
    return np.log(prob / (1 - prob + 1e-15))


def logodds2prob(logodds) -> np.ndarray:
    return 1 - 1 / (1 + np.exp(logodds) + 1e-15)


# Bresenham's Line Generation Algorithm
# https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
def bresenham(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> list[list[int]]:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    steep = 0
    if dx <= dy:
        steep = 1
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    pk = 2 * dy - dx

    loc = []
    for _ in range(0, dx + 1):
        if (
            (x1 < 0 or y1 < 0)
            or (steep == 0 and (x1 >= h or y1 >= w))
            or (steep == 1 and (x1 >= w or y1 >= h))
        ):
            break

        if steep == 0:
            loc.append([x1, y1])
        else:
            loc.append([y1, x1])

        if x1 < x2:
            x1 = x1 + 1
        else:
            x1 = x1 - 1

        if pk < 0:
            pk = pk + 2 * dy
        else:
            if y1 < y2:
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            pk = pk + 2 * dy - 2 * dx

    return loc
