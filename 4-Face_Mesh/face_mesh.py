import cv2
from cv2 import Mat
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, staticMode: bool=False, maxFaces: int=2, minDetectionCon: float=0.5, minTrackCon: float=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, 
                                                 max_num_faces=self.maxFaces, 
                                                 min_detection_confidence=self.minDetectionCon, 
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=3)

    def findFaceMesh(self, frame: Mat, draw=True):
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
        face = []
        for id,lm in enumerate(faceLms.landmark):
            #print(lm)
            ih, iw, ic = frame.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            #cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
            # 0.7, (0, 255, 0), 1)

        #print(id,x,y)
        face.append([x,y])
        faces.append(face)
        return frame, faces

def main():
    cap = cv2.VideoCapture("Media/Faces/KarolinaGrabowska.mp4")
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, frame = cap.read()
        frame, faces = detector.findFaceMesh(frame)
        if len(faces)!= 0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
        3, (0, 255, 0), 3)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

