import cv2
import mediapipe as mp
import numpy as np
import time
import queue
import threading
import pyttsx3

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed of speech

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

# Define an Exercise class to handle squats
class Squat:
    def __init__(self):
        self.counter = 0
        self.correct_counter = 0
        self.incorrect_counter = 0
        self.stage = None
        self.feedback = ""
        self.last_instruction_time = time.time()
        self.instruction_queue = queue.Queue()
        self.start_time = None
        self.squat_times = []
        self.lock = threading.Lock()

    def update(self, landmarks):
        # Extract the required landmarks
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)

        # Update stage and counter based on squat
        if angle > 160:
            self.stage = "up"
        if angle < 70 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.squat_times.append(time.time() - self.start_time if self.start_time else 0)
            self.start_time = time.time()
            self.check_form(landmarks)

        return angle

    def check_form(self, landmarks):
        # Extract landmarks
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y

        # Check for common squat mistakes
        if knee > hip and knee > ankle:
            self.correct_counter += 1
            self.feedback = "Good form! Keep it up!"
        else:
            self.incorrect_counter += 1
            if knee < ankle:
                self.feedback = "Knees should not go past your toes!"
            elif hip > shoulder:
                self.feedback = "Keep your chest up and back straight!"
            elif abs(elbow - wrist) > 0.1:
                self.feedback = "Keep your arms steady and parallel to the ground!"
            elif abs(hip - knee) < 0.1:
                self.feedback = "Lower your hips further for a deeper squat!"
            elif abs(knee - ankle) > 0.2:
                self.feedback = "Align your knees over your ankles!"
            else:
                self.feedback = "Adjust your posture for better form!"

        # Add feedback to the queue
        with self.lock:
            self.instruction_queue.put(self.feedback)

        # Remind to rest after every 5 squats
        if self.counter % 5 == 0:
            with self.lock:
                self.instruction_queue.put("Take a rest for 10 seconds. You're doing great!")

    def get_feedback(self):
        with self.lock:
            if not self.instruction_queue.empty() and time.time() - self.last_instruction_time > 2:
                self.last_instruction_time = time.time()
                return self.instruction_queue.get()
        return None

# Function to speak instructions in a separate thread
def speak_instructions(squat):
    while True:
        feedback = squat.get_feedback()
        if feedback:
            engine.say(feedback)
            engine.runAndWait()
        time.sleep(0.1)

def main():
    cap = cv2.VideoCapture(0)
    squat = Squat()

    # Start the TTS thread
    tts_thread = threading.Thread(target=speak_instructions, args=(squat,), daemon=True)
    tts_thread.start()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                angle = squat.update(landmarks)

                # Display feedback
                feedback = squat.get_feedback()
                if feedback:
                    print(feedback)

                # Display angle
                image_height, image_width, _ = image.shape
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                hip_pixel = tuple(np.multiply(hip, [image_width, image_height]).astype(int))
                knee_pixel = tuple(np.multiply(knee, [image_width, image_height]).astype(int))
                ankle_pixel = tuple(np.multiply(ankle, [image_width, image_height]).astype(int))

                cv2.putText(image, str(int(angle)), knee_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Render rep counter and stage
                cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
                cv2.putText(image, "REPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(squat.counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "STAGE", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, squat.stage if squat.stage else "", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, squat.feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            except AttributeError:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow("AI Trainer", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Calculate average squat time
        if squat.squat_times:
            average_squat_time = np.mean(squat.squat_times)
            print(f"Average time per squat: {average_squat_time:.2f} seconds")
        print(f"Total squats: {squat.counter}")
        print(f"Correct squats: {squat.correct_counter}")
        print(f"Incorrect squats: {squat.incorrect_counter}")

if __name__ == "__main__":
    main()