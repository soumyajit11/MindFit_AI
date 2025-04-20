import cv2
from cvzone.PoseModule import PoseDetector
import math
import numpy as np

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

    # finding angles
    def angle(self):
        if len(self.lmlist) != 0:
            point1 = self.lmlist[self.p1]
            point2 = self.lmlist[self.p2]
            point3 = self.lmlist[self.p3]
            point4 = self.lmlist[self.p4]
            point5 = self.lmlist[self.p5]
            point6 = self.lmlist[self.p6]

            if len(point1) >= 2 and len(point2) >= 2 and len(point3) >= 2 and len(point4) >= 2 and len(point5) >= 2 and len(
                    point6) >= 2:
                x1, y1 = point1[:2]
                x2, y2 = point2[:2]
                x3, y3 = point3[:2]
                x4, y4 = point4[:2]
                x5, y5 = point5[:2]
                x6, y6 = point6[:2]

                # calculating angle for left and right hands
                leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                # drawing circles and lines on selected points
                if self.drawPoints:
                    cv2.circle(img, (x1, y1), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), 6)
                    cv2.circle(img, (x2, y2), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x2, y2), 15, (0, 255, 0), 6)
                    cv2.circle(img, (x3, y3), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x3, y3), 15, (0, 255, 0), 6)
                    cv2.circle(img, (x4, y4), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x4, y4), 15, (0, 255, 0), 6)
                    cv2.circle(img, (x5, y5), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x5, y5), 15, (0, 255, 0), 6)
                    cv2.circle(img, (x6, y6), 10, (0, 255, 255), 5)
                    cv2.circle(img, (x6, y6), 15, (0, 255, 0), 6)

                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                    cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                    cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                    cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 4)

                return [leftHandAngle, rightHandAngle]

# AI-Generated Feedback System
def provide_feedback(left, right, lmList):
    feedback = []

    # Feedback 1: Arm Position
    if left < 90 or right < 90:
        feedback.append(f"Raise your arms higher! You need to bend {90 - max(left, right)}° more.")

    # Feedback 2: Back Straightness
    if len(lmList) >= 24:
        shoulder_left = lmList[11]
        shoulder_right = lmList[12]
        hip_left = lmList[23]
        hip_right = lmList[24]

        if shoulder_left and shoulder_right and hip_left and hip_right:
            back_angle = math.degrees(math.atan2(hip_left[1] - shoulder_left[1], hip_left[0] - shoulder_left[0]) - 
                                      math.atan2(hip_right[1] - shoulder_right[1], hip_right[0] - shoulder_right[0]))
            if abs(back_angle) > 10:
                feedback.append("Keep your back straight! Avoid leaning to one side.")

    # Feedback 3: Elbow-Body Angle
    if len(lmList) >= 24:
        elbow_left = lmList[13]
        elbow_right = lmList[14]
        shoulder_left = lmList[11]
        shoulder_right = lmList[12]
        hip_left = lmList[23]
        hip_right = lmList[24]

        if elbow_left and elbow_right and shoulder_left and shoulder_right and hip_left and hip_right:
            # Calculate elbow-body angle for left and right arms
            elbow_body_angle_left = math.degrees(math.atan2(shoulder_left[1] - elbow_left[1], shoulder_left[0] - elbow_left[0]) - 
                                                 math.atan2(hip_left[1] - elbow_left[1], hip_left[0] - elbow_left[0]))
            elbow_body_angle_right = math.degrees(math.atan2(shoulder_right[1] - elbow_right[1], shoulder_right[0] - elbow_right[0]) - 
                                                  math.atan2(hip_right[1] - elbow_right[1], hip_right[0] - elbow_right[0]))

            # Display elbow-body angle on the screen
            cv2.putText(img, f"Left Elbow-Body Angle: {int(abs(elbow_body_angle_left))}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Right Elbow-Body Angle: {int(abs(elbow_body_angle_right))}°", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Provide feedback if the angle is not within the desired range
            desired_angle = 90  # Desired elbow-body angle
            if abs(elbow_body_angle_left) < desired_angle:
                feedback.append(f"Bend your left elbow more! You need to bend {desired_angle - int(abs(elbow_body_angle_left))}° more.")
            if abs(elbow_body_angle_right) < desired_angle:
                feedback.append(f"Bend your right elbow more! You need to bend {desired_angle - int(abs(elbow_body_angle_right))}° more.")

    # Feedback 4: Overall Posture
    if len(feedback) == 0:
        feedback.append("Great posture! Keep it up!")

    return feedback

# Main Loop
counter = 0
direction = 0

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (640, 480))

    detector.findPose(img, draw=0)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=0, draw=False)

    angle1 = angleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
    hands = angle1.angle()
    if hands is not None:
        left, right = hands[0:]
    else:
        left, right = 0, 0

    # Counting number of shoulder ups
    if left >= 90 and right >= 90:
        if direction == 0:
            counter += 0.5
            direction = 1
    if left <= 70 and right <= 70:
        if direction == 1:
            counter += 0.5
            direction = 0

    # Getting feedback from the instructor
    feedback = provide_feedback(left, right, lmList)

    # Displaying feedback on the screen
    y_offset = 120
    for line in feedback:
        cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    # Putting scores on the screen
    cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
    cv2.putText(img, str(int(counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)