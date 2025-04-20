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

    def update(self, landmarks, angle):
        # Check squat stage
        if angle > 160:  # Standing position
            if self.stage == "down":
                self.stage = "up"
                self.counter += 1
                feedback_queue.put(f"Good job! You've completed {self.counter} squats.")
            if self.stage is None:
                self.stage = "up"

        if angle < 70:  # Squat position
            if self.stage == "up":
                self.stage = "down"
                feedback_queue.put("Lower your body until your thighs are parallel to the floor.")
            if self.stage is None:
                self.stage = "down"

        # Provide feedback on form
        if 70 <= angle <= 160:  # Intermediate position
            feedback_queue.put("Keep your chest up and back straight.")

        return self.counter, self.stage

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

                # Calculate knee angle
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                # Update squat trainer
                counter, stage = squat_trainer.update(landmarks, angle)

                # Display angle
                cv2.putText(
                    image,
                    f"Knee Angle: {int(angle)}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Display rep counter and stage
                cv2.putText(
                    image,
                    f"Reps: {counter}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Stage: {stage}",
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