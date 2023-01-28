import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, mode = False, max_hands = 2, detection_conf = 0.5, tracking_conf = 0.5):
        
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.detection_conf, self.tracking_conf)
        self.mpdraw = mp.solutions.drawing_utils


    def find_hands(self,img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, handno = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0, 255), cv2.FILLED)
        return lmlist




