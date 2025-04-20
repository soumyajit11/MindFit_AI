import cv2
import mediapipe as mp
from PoseModule import PoseDetectorModified
from Exercise import Exercise

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Exercise instructions
exercise_instructions = {
    "Bicep Curl": [
        "Stand straight with a dumbbell in each hand.",
        "Keep your elbows close to your torso.",
        "Curl the weights while contracting your biceps.",
        "Slowly lower the dumbbells back to the starting position."
    ],
    "Squat": [
        "Stand with your feet shoulder-width apart.",
        "Lower your body until your thighs are parallel to the floor.",
        "Keep your chest up and your back straight.",
        "Push through your heels to return to the starting position."
    ],
    "Push-Up": [
        "Start in a plank position with your hands slightly wider than shoulder-width apart.",
        "Lower your body until your chest nearly touches the floor.",
        "Keep your body in a straight line from head to toe.",
        "Push yourself back up to the starting position."
    ]
}

def check_form(exercise, landmarks):
    if exercise == "Squat":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        angle = Exercise.calculate_angle(None, shoulder, hip, knee)

        if angle < 160:
            return "Keep your back straight!"
    
    elif exercise == "Bicep Curl":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = Exercise.calculate_angle(None, shoulder, elbow, wrist)

        if angle > 30:
            return "Keep your elbows close to your torso!"

    return None

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetectorModified()

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

    current_exercise = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = detector.findPose(frame)
        landmarks_list = detector.findPosition(image, draw=False)

        if landmarks_list:
            if current_exercise:
                angle = exercises[current_exercise].update(landmarks_list)
                feedback = check_form(current_exercise, landmarks_list)

                if feedback:
                    cv2.putText(image, feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display instructions
                for i, instruction in enumerate(exercise_instructions[current_exercise]):
                    cv2.putText(image, instruction, (10, 180 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("AI Trainer", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()