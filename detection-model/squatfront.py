import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
import queue

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Global variables for feedback and threading
feedback_queue = queue.Queue()  # Thread-safe queue for feedback messages
exit_flag = False  # Flag to gracefully exit the program

# Function to speak feedback in a separate thread
def speak_feedback():
    global feedback_queue, exit_flag
    while not exit_flag:
        if not feedback_queue.empty():
            feedback = feedback_queue.get()
            if feedback:
                engine.say(feedback)
                engine.runAndWait()
        time.sleep(0.1)  # Small delay to prevent high CPU usage

# Start the feedback thread
feedback_thread = threading.Thread(target=speak_feedback, daemon=True)
feedback_thread.start()

# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # Calculate the angle in radians
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is within [0, 180]
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Squat Trainer Class
class SquatTrainer:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.start_time = None
        self.squat_time = None

    def update(self, landmarks, hip_y, knee_y):
        # Check squat stage based on vertical movement of hips and knees
        if hip_y < knee_y:  # Standing position
            if self.stage == "down":
                self.stage = "up"
                self.counter += 1
                self.squat_time = time.time() - self.start_time
                feedback_queue.put(f"Squat {self.counter} completed in {self.squat_time:.2f} seconds.")
                print(f"Squat {self.counter} completed in {self.squat_time:.2f} seconds.")
            if self.stage is None:
                self.stage = "up"
                self.start_time = time.time()

        if hip_y > knee_y:  # Squat position
            if self.stage == "up":
                self.stage = "down"
                feedback_queue.put("Move up.")
            if self.stage is None:
                self.stage = "down"

        return self.counter, self.stage, self.squat_time

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Initialize Squat Trainer
        squat_trainer = SquatTrainer()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get hip and knee landmarks
                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

                # Update squat trainer based on vertical position of hips and knees
                counter, stage, squat_time = squat_trainer.update(landmarks, hip.y, knee.y)

                # Display rep counter and stage
                cv2.putText(
                    image,
                    f"Reps: {counter}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Stage: {stage}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Display squat time
                if squat_time is not None:
                    cv2.putText(
                        image,
                        f"Time: {squat_time:.2f} sec",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            except AttributeError:
                pass

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

            # Display the resulting image
            cv2.imshow("Squat Trainer", image)

            # Handle keypresses
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                # Quit the application
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()