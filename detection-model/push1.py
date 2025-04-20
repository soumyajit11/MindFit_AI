import cv2
import mediapipe as mp
import math
from gtts import gTTS
import pygame
import os
import time

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def play_speech(text, gif_path):
    # Generate speech using gTTS
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")

    # Initialize pygame for audio and GIF
    pygame.mixer.init()
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.play()

    # Load and display GIF
    cap_gif = cv2.VideoCapture(gif_path)
    while pygame.mixer.music.get_busy():
        ret, frame = cap_gif.read()
        if not ret:
            cap_gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.imshow("Talking Trainer", frame)
        cv2.waitKey(30)

    # Release the GIF capture
    cap_gif.release()

    # Stop and unload the music to release the file
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    # Delete the temporary MP3 file
    while os.path.exists("temp.mp3"):
        try:
            os.remove("temp.mp3")
        except PermissionError:
            time.sleep(0.1)  # Wait and retry if the file is still locked


def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture('squat_recording.mp4')

    # Pushup counter variables
    counter = 0
    direction = 0  # 0: going down, 1: going up

    # Path to the talking GIF
    gif_path = "kk.gif"

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Calculate angle between shoulder, elbow, and wrist
            angle = detector.findAngle(img, 11, 13, 15, draw=False)

            # Pushup counter logic
            if angle > 160:
                direction = 0  # Going down
            if angle < 70 and direction == 0:
                direction = 1  # Going up
                counter += 1
                print(f"Pushup Count: {counter}")

                # Speak the count
                play_speech(f"{counter}", gif_path)

                # After every 5 pushups, suggest rest
                if counter % 5 == 0:
                    play_speech("Take a rest. Stretch your muscles.", gif_path)

            # Display pushup count on the screen
            cv2.putText(img, f"Pushups: {counter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imshow('Pushup Counter', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()