import time
import cv2
from hand_detector import HandDetector


prev_time = 0
curr_time = 0
cap = cv2.VideoCapture(0)
detector = HandDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist[4])

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (10, 40), 
    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)