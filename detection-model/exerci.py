import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  # Text-to-speech library
import threading  # For running TTS in a separate thread

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed of speech

# Function to run TTS in a separate thread
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define a function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Define an Exercise class to handle different exercises
class Exercise:
    def __init__(self, name, landmarks, angle_points, up_threshold, down_threshold):
        self.name = name
        self.landmarks = landmarks
        self.angle_points = angle_points
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.counter = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.stage = None

    def update(self, landmarks):
        # Extract the required landmarks
        a = [landmarks[self.angle_points[0]].x, landmarks[self.angle_points[0]].y]
        b = [landmarks[self.angle_points[1]].x, landmarks[self.angle_points[1]].y]
        c = [landmarks[self.angle_points[2]].x, landmarks[self.angle_points[2]].y]

        # Calculate angle
        angle = calculate_angle(a, b, c)

        # Update stage and counter based on exercise type
        if self.name == "Bicep Curl":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                if 30 <= angle <= 160:  # Correct form
                    self.correct_reps += 1
                    threading.Thread(target=speak, args=("Good job! That's a correct rep.",)).start()
                else:  # Incorrect form
                    self.incorrect_reps += 1
                    threading.Thread(target=speak, args=("Incorrect form. Bend your elbow more.",)).start()
                print(f"{self.name} Reps: {self.counter}")

        elif self.name == "Squat":
            if angle > self.up_threshold:
                self.stage = "up"
            if angle < self.down_threshold and self.stage == "up":
                self.stage = "down"
                self.counter += 1
                if 70 <= angle <= 160:  # Correct form
                    self.correct_reps += 1
                    threading.Thread(target=speak, args=("Well done! That's a correct squat.",)).start()
                else:  # Incorrect form
                    self.incorrect_reps += 1
                    threading.Thread(target=speak, args=("Incorrect form. Bend your knees more.",)).start()
                print(f"{self.name} Reps: {self.counter}")

        elif self.name == "Push-Up":
            if angle > self.up_threshold:
                self.stage = "down"
            if angle < self.down_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                if 90 <= angle <= 160:  # Correct form
                    self.correct_reps += 1
                    threading.Thread(target=speak, args=("Great! That's a correct push-up.",)).start()
                else:  # Incorrect form
                    self.incorrect_reps += 1
                    threading.Thread(target=speak, args=("Incorrect form. Lower your body more.",)).start()
                print(f"{self.name} Reps: {self.counter}")

        return angle


# Function to detect current exercise based on angles
def detect_exercise(angles, prev_exercise, angle_changes):
    detected_exercise = None
    angle_change_threshold = 10  # degrees

    if (
        abs(angle_changes["bicep_curl"]) > angle_change_threshold
        and abs(angle_changes["squat_knee"]) < angle_change_threshold
        and abs(angle_changes["pushup_elbow"]) < angle_change_threshold
    ):
        detected_exercise = "Bicep Curl"
    elif (
        abs(angle_changes["squat_knee"]) > angle_change_threshold
        and abs(angle_changes["bicep_curl"]) < angle_change_threshold
        and abs(angle_changes["pushup_elbow"]) < angle_change_threshold
    ):
        detected_exercise = "Squat"
    elif (
        abs(angle_changes["pushup_elbow"]) > angle_change_threshold
        and abs(angle_changes["bicep_curl"]) < angle_change_threshold
        and abs(angle_changes["squat_knee"]) < angle_change_threshold
    ):
        detected_exercise = "Push-Up"

    return detected_exercise


def main():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        exercises = {
            "Bicep Curl": Exercise(
                name="Bicep Curl",
                landmarks=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                ),
                up_threshold=160,
                down_threshold=30,
            ),
            "Squat": Exercise(
                name="Squat",
                landmarks=("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value,
                ),
                up_threshold=160,
                down_threshold=70,
            ),
            "Push-Up": Exercise(
                name="Push-Up",
                landmarks=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
                angle_points=(
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                ),
                up_threshold=160,
                down_threshold=90,
            ),
        }

        prev_angles = {"bicep_curl": 0, "squat_knee": 0, "pushup_elbow": 0}
        current_exercise = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                angles = {
                    "bicep_curl": calculate_angle(
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[0]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[1]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Bicep Curl"].angle_points[2]].x,
                            landmarks[exercises["Bicep Curl"].angle_points[2]].y,
                        ],
                    ),
                    "squat_knee": calculate_angle(
                        [
                            landmarks[exercises["Squat"].angle_points[0]].x,
                            landmarks[exercises["Squat"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Squat"].angle_points[1]].x,
                            landmarks[exercises["Squat"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Squat"].angle_points[2]].x,
                            landmarks[exercises["Squat"].angle_points[2]].y,
                        ],
                    ),
                    "pushup_elbow": calculate_angle(
                        [
                            landmarks[exercises["Push-Up"].angle_points[0]].x,
                            landmarks[exercises["Push-Up"].angle_points[0]].y,
                        ],
                        [
                            landmarks[exercises["Push-Up"].angle_points[1]].x,
                            landmarks[exercises["Push-Up"].angle_points[1]].y,
                        ],
                        [
                            landmarks[exercises["Push-Up"].angle_points[2]].x,
                            landmarks[exercises["Push-Up"].angle_points[2]].y,
                        ],
                    ),
                }

                angle_changes = {
                    "bicep_curl": angles["bicep_curl"] - prev_angles["bicep_curl"],
                    "squat_knee": angles["squat_knee"] - prev_angles["squat_knee"],
                    "pushup_elbow": angles["pushup_elbow"] - prev_angles["pushup_elbow"],
                }

                prev_angles = angles.copy()

                detected_exercise = detect_exercise(angles, current_exercise, angle_changes)

                if detected_exercise:
                    current_exercise = detected_exercise
                    threading.Thread(target=speak, args=(f"Starting {current_exercise}. Let's go!",)).start()

                if current_exercise:
                    angle = exercises[current_exercise].update(landmarks)

                    # Display angle and feedback on the screen
                    cv2.putText(
                        image,
                        f"Angle: {int(angle)}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        f"Correct Reps: {exercises[current_exercise].correct_reps}",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        f"Incorrect Reps: {exercises[current_exercise].incorrect_reps}",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            except AttributeError:
                pass

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

            cv2.imshow("Personal Trainer", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()