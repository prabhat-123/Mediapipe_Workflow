import os
import cv2
import sys
import time
import mediapipe as mp

# sys.path.append(os.path.dirname(os.getcwd()))
# print(sys.path)
mpdraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose = mppose.Pose()

cap = cv2.VideoCapture('anup_gim.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('anup_gim_pose_estimation.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, size)
prev_time = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpdraw.draw_landmarks(img, results.pose_landmarks, mppose.POSE_CONNECTIONS,
        mpdraw.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=2),
        mpdraw.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=2))
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx , cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (10, 40), 
    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    result.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(2)