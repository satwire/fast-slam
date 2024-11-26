import json

import cv2


class Scene:
    def __init__(self):
        self.__x = 0
        self.__y = 0
        self.__theta = 0
        self.movements = []

        self.raw_map = None
        self.map = None
        self.height = 0
        self.width = 0

        self.__load_map()
        self.__load_scene()

    def __load_map(self) -> None:
        self.raw_map = cv2.imread("scene.png")
        gray = cv2.cvtColor(self.raw_map, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
        _, gray = cv2.threshold(gray, 254, 1, cv2.THRESH_BINARY)
        self.map = gray
        self.height, self.width = gray.shape[:2]

    def __load_scene(self) -> None:
        with open("scene.json") as f:
            data = json.load(f)
            self.movements = data["movements"]

            initials = data["initial"]
            self.__x = initials["x"]
            self.__y = initials["y"]
            self.__theta = initials["theta"]

    def get_initial_state(self) -> tuple[int, int, int]:
        return self.__x, self.__y, self.__theta
