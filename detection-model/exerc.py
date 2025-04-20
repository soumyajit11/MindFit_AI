import cv2
from cvzone.PoseModule import PoseDetector
import math
import numpy as np
from gtts import gTTS
import os

cap = cv2.VideoCapture(0)
detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

# Creating Angle finder class
class angleFinder:
    def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawPoints):
        self.lmlist = lmlist
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.drawPoints = drawPoints

    def angle(self):
        if len(self.lmlist) != 0:
            point1 = self.lmlist[self.p1]
            point2 = self.lmlist[self.p2]
            point3 = self.lmlist[self.p3]
            point4 = self.lmlist[self.p4]
            point5 = self.lmlist[self.p5]
            point6 = self.lmlist[self.p6]

            if all(len(pt) >= 2 for pt in [point1, point2, point3, point4, point5, point6]):
                x1, y1 = point1[:2]
                x2, y2 = point2[:2]
                x3, y3 = point3[:2]
                x4, y4 = point4[:2]
                x5, y5 = point5[:2]
                x6, y6 = point6[:2]

                leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                return [leftHandAngle, rightHandAngle]
        return None

def provide_feedback(left, right, lmList):
    feedback = []
    warning = None

    if left < 20 or right < 20:
        warning = "Warning! Lift your hands higher."

    if left < 90 or right < 90:
        feedback.append(f"Raise your arms higher! You need to bend {90 - max(left, right)}° more.")

    if len(lmList) >= 24:
        shoulder_left, shoulder_right = lmList[11], lmList[12]
        hip_left, hip_right = lmList[23], lmList[24]

        if all(shoulder_left) and all(shoulder_right) and all(hip_left) and all(hip_right):
            back_angle = math.degrees(math.atan2(hip_left[1] - shoulder_left[1], hip_left[0] - shoulder_left[0]) - 
                                      math.atan2(hip_right[1] - shoulder_right[1], hip_right[0] - shoulder_right[0]))
            if abs(back_angle) > 10:
                feedback.append("Keep your back straight! Avoid leaning to one side.")

    if warning:
        tts = gTTS(text=warning, lang='en')
        tts.save("warning.mp3")
        os.system("mpg321 warning.mp3")  # Requires mpg321 installed

    return feedback

counter = 0
direction = 0

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (640, 480))

    detector.findPose(img, draw=0)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=0, draw=False)

    angle1 = angleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
    hands = angle1.angle()
    
    if hands:
        left, right = hands
    else:
        left, right = 0, 0

    if left >= 90 and right >= 90:
        if direction == 0:
            counter += 0.5
            direction = 1
    if left <= 70 and right <= 70:
        if direction == 1:
            counter += 0.5
            direction = 0

    feedback = provide_feedback(left, right, lmList)

    y_offset = 120
    for line in feedback:
        cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
    cv2.putText(img, str(int(counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
