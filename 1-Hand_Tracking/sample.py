import cv2
import mediapipe as mp
import time
import hand_tracking as ht

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = ht.HandDetector()
while True:
    _, frame = cap.read()
    frame = detector.findHands(frame, draw=True )
    landmarks = detector.findPosition(frame, draw=False)
    if len(landmarks) != 0:
        print(landmarks[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
     #           (255, 0, 255), 3)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)