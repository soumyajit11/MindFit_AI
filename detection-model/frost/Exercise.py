import numpy as np

class Exercise:
    def __init__(self, name, landmarks, angle_points, up_threshold, down_threshold):
        self.name = name
        self.landmarks = landmarks
        self.angle_points = angle_points
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.counter = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def update(self, landmarks):
        a = [landmarks[self.angle_points[0]].x, landmarks[self.angle_points[0]].y]
        b = [landmarks[self.angle_points[1]].x, landmarks[self.angle_points[1]].y]
        c = [landmarks[self.angle_points[2]].x, landmarks[self.angle_points[2]].y]

        angle = self.calculate_angle(a, b, c)

        if self.name == "Bicep Curl":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1

        elif self.name == "Squat":
            if angle > self.up_threshold:
                self.stage = "up"
            if angle < self.down_threshold and self.stage == "up":
                self.stage = "down"
                self.counter += 1

        elif self.name == "Push-Up":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1

        return angle