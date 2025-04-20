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

# Instructor feedback function
def provide_feedback(left, right, lmList):
    feedback = []
    
    # Feedback based on arm angles
    if left >= 90 and right >= 90:
        feedback.append("Good form! Keep it up!")
    elif left < 90 or right < 90:
        feedback.append("Raise your arms higher!")
    
    # Feedback based on back straightness
    if len(lmList) >= 24:  # Assuming lmList has at least 24 points (full body)
        shoulder_left = lmList[11]
        shoulder_right = lmList[12]
        hip_left = lmList[23]
        hip_right = lmList[24]
        
        if shoulder_left and shoulder_right and hip_left and hip_right:
            back_angle = math.degrees(math.atan2(hip_left[1] - shoulder_left[1], hip_left[0] - shoulder_left[0]) - 
                                      math.atan2(hip_right[1] - shoulder_right[1], hip_right[0] - shoulder_right[0]))
            if abs(back_angle) > 10:
                feedback.append("Keep your back straight!")
    
    # Feedback based on lower body position
    if len(lmList) >= 24:
        knee_left = lmList[25]
        knee_right = lmList[26]
        ankle_left = lmList[27]
        ankle_right = lmList[28]
        
        if knee_left and knee_right and ankle_left and ankle_right:
            knee_angle_left = math.degrees(math.atan2(ankle_left[1] - knee_left[1], ankle_left[0] - knee_left[0]) - 
                                           math.atan2(hip_left[1] - knee_left[1], hip_left[0] - knee_left[0]))
            knee_angle_right = math.degrees(math.atan2(ankle_right[1] - knee_right[1], ankle_right[0] - knee_right[0]) - 
                                            math.atan2(hip_right[1] - knee_right[1], hip_right[0] - knee_right[0]))
            
            if knee_angle_left < 160 or knee_angle_right < 160:
                feedback.append("Bend your knees slightly!")
    
    # Feedback based on head position
    if len(lmList) >= 24:
        nose = lmList[0]
        shoulder_left = lmList[11]
        shoulder_right = lmList[12]
        
        if nose and shoulder_left and shoulder_right:
            head_angle = math.degrees(math.atan2(shoulder_left[1] - nose[1], shoulder_left[0] - nose[0]) - 
                                      math.atan2(shoulder_right[1] - nose[1], shoulder_right[0] - nose[0]))
            if abs(head_angle) > 10:
                feedback.append("Keep your head straight!")
    
    # Feedback based on elbow position
    if len(lmList) >= 24:
        elbow_left = lmList[13]
        elbow_right = lmList[14]
        
        if elbow_left and elbow_right:
            elbow_angle_left = math.degrees(math.atan2(shoulder_left[1] - elbow_left[1], shoulder_left[0] - elbow_left[0]) - 
                                            math.atan2(knee_left[1] - elbow_left[1], knee_left[0] - elbow_left[0]))
            elbow_angle_right = math.degrees(math.atan2(shoulder_right[1] - elbow_right[1], shoulder_right[0] - elbow_right[0]) - 
                                             math.atan2(knee_right[1] - elbow_right[1], knee_right[0] - elbow_right[0]))
            
            if elbow_angle_left < 90 or elbow_angle_right < 90:
                feedback.append("Keep your elbows up!")
    
    # Feedback based on foot position
    if len(lmList) >= 24:
        foot_left = lmList[27]
        foot_right = lmList[28]
        
        if foot_left and foot_right:
            foot_angle_left = math.degrees(math.atan2(knee_left[1] - foot_left[1], knee_left[0] - foot_left[0]) - 
                                           math.atan2(ankle_left[1] - foot_left[1], ankle_left[0] - foot_left[0]))
            foot_angle_right = math.degrees(math.atan2(knee_right[1] - foot_right[1], knee_right[0] - foot_right[0]) - 
                                            math.atan2(ankle_right[1] - foot_right[1], ankle_right[0] - foot_right[0]))
            
            if foot_angle_left < 80 or foot_angle_right < 80:
                feedback.append("Keep your feet flat on the ground!")
    
    # Feedback based on hip position
    if len(lmList) >= 24:
        hip_angle = math.degrees(math.atan2(shoulder_left[1] - hip_left[1], shoulder_left[0] - hip_left[0]) - 
                                 math.atan2(shoulder_right[1] - hip_right[1], shoulder_right[0] - hip_right[0]))
        if abs(hip_angle) > 10:
            feedback.append("Keep your hips level!")
    
    # Feedback based on shoulder position
    if len(lmList) >= 24:
        shoulder_angle = math.degrees(math.atan2(hip_left[1] - shoulder_left[1], hip_left[0] - shoulder_left[0]) - 
                                      math.atan2(hip_right[1] - shoulder_right[1], hip_right[0] - shoulder_right[0]))
        if abs(shoulder_angle) > 10:
            feedback.append("Keep your shoulders level!")
    
    # Feedback based on overall posture
    if len(feedback) == 0:
        feedback.append("Great posture! Keep it up!")
    
    return feedback

# defining some variables
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
    y_offset = 50
    for line in feedback:
        cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    # putting scores on the screen
    cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
    cv2.putText(img, str(int(counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

    # Converting values for rectangles
    leftval = np.interp(left, [0, 100], [400, 200])
    rightval = np.interp(right, [0, 100], [400, 200])

    # For color changing
    value_left = np.interp(left, [0, 100], [0, 100])
    value_right = np.interp(right, [0, 100], [0, 100])

    # Drawing right rectangle and putting text
    cv2.putText(img, 'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
    cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
    cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

    # Drawing left rectangle and putting text
    cv2.putText(img, 'L', (604, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
    cv2.rectangle(img, (582, 200), (632, 400), (0, 255, 0), 5)
    cv2.rectangle(img, (582, int(leftval)), (632, 400), (255, 0, 0), -1)

    if value_left > 70:
        cv2.rectangle(img, (582, int(leftval)), (632, 400), (0, 0, 255), -1)

    if value_right > 70:
        cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)