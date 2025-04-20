import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed of speech

# Push-up counter
count = 0
position = None  # 'up' or 'down'
last_feedback_time = time.time()

# Function to give voice feedback
def speak_feedback(text):
    global last_feedback_time
    if time.time() - last_feedback_time > 2:  # Avoid overlapping speech
        engine.say(text)
        engine.runAndWait()
        last_feedback_time = time.time()

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        
        # Get key points
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

        # Check push-up position
        if elbow > shoulder and elbow > wrist:  # Down position
            if position != 'down':
                position = 'down'
        elif elbow < shoulder:  # Up position
            if position == 'down':
                count += 1
                position = 'up'
                speak_feedback(f"Good job! Push-up count: {count}")
                if count % 5 == 0:
                    speak_feedback("Great work! Keep going!")
        
        # Form corrections
        if hip < shoulder:  # Hips too high
            speak_feedback("Lower your hips for proper form.")
        elif hip > shoulder * 1.2:  # Hips too low
            speak_feedback("Keep your body straight.")
    
    # Display push-up count
    cv2.putText(frame, f'Push-ups: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Push-up Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
