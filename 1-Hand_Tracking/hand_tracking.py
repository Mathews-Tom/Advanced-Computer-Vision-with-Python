import cv2
from cv2 import Mat
import mediapipe as mp
import time


class HandDetector():
    """HandDetector class"""
    def __init__(self, image_mode: bool=False, max_hands: int=2, detection_conf: float=0.5, 
                    tracking_conf: float=0.5):
        """Initialize the HandDetector.

        Args:
            image_mode (bool, optional): Image mode is static or live. Defaults to False.
            max_hands (int, optional): Maximum number of hands. Defaults to 2.
            detection_conf (float, optional): Detenction confidence. Defaults to 0.5.
            tracking_conf (float, optional): Tracking confidence. Defaults to 0.5.
        """
        self.image_mode = image_mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.image_mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_conf, 
                                        min_tracking_confidence=self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils
    
    def findHands(self, img: Mat, draw: bool=True) -> Mat:
        """Find hands in the given image.

        Args:
            img (Mat): Image to find hands in.
            draw (bool, optional): Draw hand on the image. Defaults to True.

        Returns:
            Mat: Image with hands overlayed on the given image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, 
                                                handLms,
                                                self.mp_hands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img: Mat, hand_no: int=0, draw: bool=True )-> list:
        """Find landmark positions in the given image.

        Args:
            img (Mat): Image with hands found.
            hand_no (int, optional): Hands index. Defaults to 0.
            draw (bool, optional): Draw landmarks on the image. Defaults to True.

        Returns:
            list: Image with hands and landmarks overlayed on the given image.
        """
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return landmarks_list


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        landmarks = detector.findPosition(img)

        if len(landmarks) != 0:
            print(landmarks[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
