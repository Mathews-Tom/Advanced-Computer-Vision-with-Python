import cv2
from cv2 import Mat
import mediapipe as mp
import time
import math
class PoseDetector():
    """Pose detector class"""
    def __init__(self, image_mode: bool=False,smooth: bool=True, 
                 detection_conf: float=0.5, track_conf: float=0.5):
        """Initialize the PoseDetector.

        Args:
            image_mode (bool, optional): Image frame mode is static or live.. Defaults to False.
            smooth (bool, optional): Smooth landmarks in pose. Defaults to True.
            detection_conf (float, optional): Detection confidence. Defaults to 0.5.
            track_conf (float, optional): Tracking confidence. Defaults to 0.5.
        """
        self.image_mode = image_mode
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.image_mode, 
                                    #  upper_body_only=self.upper_body, 
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detection_conf, 
                                     min_tracking_confidence=self.track_conf)
    
    def findPose(self, frame: Mat, draw: bool=True) -> Mat:
        """Find pose for given frame.

        Args:
            frame (Mat): Frame to find a pose in.
            draw (bool, optional): Draw pose on frame. Defaults to True.

        Returns:
            Mat: Frame with pose found.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame
    
    def findPosition(self, frame: Mat, draw: bool=True) -> list:
        """Find landmark positions in the given frame 

        Args:
            frame (Mat): _description_
            draw (bool, optional): Draw landmark positions. Defaults to True.

        Returns:
            list: Landmark positions in the given frame.
        """
        self.landmarks_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks_list.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.landmarks_list
    
    def findAngle(self, frame: Mat, p1, p2, p3, draw: bool=True):
        # Get the landmarks
        x1, y1 = self.landmarks_list[p1][1:]
        x2, y2 = self.landmarks_list[p2][1:]
        x3, y3 = self.landmarks_list[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # print(angle)
        # Draw
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    cap = cv2.VideoCapture("PoseVideos/PolinaTankilevitch.mp4")
    pTime = 0
    detector = PoseDetector()
    while True:
        _, frame = cap.read()
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
