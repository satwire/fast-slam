import cv2
import numpy as np


def draw_true_path(robot, radar_list, step: int) -> None:
    img = cv2.cvtColor(((1 - robot.grid) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in radar_list:
        cv2.line(
            img,
            (int(robot.trajectory[-1][0]), int(robot.trajectory[-1][1])),
            (x, y),
            (0, 128, 128),
            1,
        )

    for i in range(len(robot.trajectory) - 1):
        p1 = robot.trajectory[i]
        p2 = robot.trajectory[i + 1]
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (255, 0, 0),
            2,
        )

    cv2.circle(
        img,
        (int(robot.trajectory[-1][0]), int(robot.trajectory[-1][1])),
        4,
        (255, 0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )

    filename = f"results/ground_truth_step_{step}.jpg"
    cv2.imwrite(filename, img)


def draw_estimated_path(best_particle, particles, step: int) -> None:
    img = cv2.cvtColor(
        ((1 - best_particle.grid) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    for i in range(len(best_particle.trajectory) - 1):
        p1 = best_particle.trajectory[i]
        p2 = best_particle.trajectory[i + 1]
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 0, 255),
            2,
        )

    cv2.circle(
        img,
        (int(best_particle.trajectory[-1][0]), int(best_particle.trajectory[-1][1])),
        4,
        (0, 0, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    for p in particles:
        cv2.circle(
            img,
            (int(p.x), int(p.y)),
            1,
            (255, 0, 0),
            -1,
        )

    filename = f"results/estimated_step_{step}.jpg"
    cv2.imwrite(filename, img)


def draw(robot, step: int, radar_list, particles, best_particle) -> None:
    draw_true_path(robot, radar_list, step)
    draw_estimated_path(best_particle, particles, step)
