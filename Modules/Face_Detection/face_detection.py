import cv2
from cv2 import Mat
import mediapipe as mp
import time
from typing import Tuple

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, frame: Mat, draw: bool=True) -> Tuple[Mat, Tuple[int, int, int, int]]:
        """Find all faces in the given frame.

        Args:
            frame (Mat): Frame
            draw (bool, optional): Draw BoundingBox or not. Defaults to True.

        Returns:
            Tuple[Mat, Tuple[int, int, int, int]]:  Frame and BoundingBox
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                        int(bboxC.width * iw), int(bboxC.height * ih))
                bboxs.append([id, bbox, detection.score])
                if draw:
                    frame = self.fancyDraw(frame,bbox)

                    cv2.putText(frame, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return frame, bboxs

    def fancyDraw(self, frame: Mat, bbox: Tuple[int, int, int, int], l: int=30, t: int=5, rt: int= 1) -> Mat:
        """Draw a fancy BoundingBox on a frame."""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(frame, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(frame, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(frame, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(frame, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(frame, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(frame, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(frame, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(frame, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return frame


def main():
    cap = cv2.VideoCapture("Media/Faces/KetutSubiyanto.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, frame = cap.read()
        frame, bboxs = detector.findFaces(frame)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

