import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
from google.protobuf.json_format import MessageToDict
import pandas as pd

class second(Thread):
        
        results=None
        arr=np.zeros((5,4,3))
        base=np.zeros((1,3))
        angles=None
        angles_t=None
        theta=np.zeros((15,1)) 
        score=0 
        bend=0

        result=None
        
        def __init__(self):
            super(second,self).__init__()
            pass

        def run(self):
            cap = cv2.VideoCapture(2)

            mpHands = mp.solutions.hands
            hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mpDraw = mp.solutions.drawing_utils

            while True:
                success, img = cap.read()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                second.results = hands.process(imgRGB)
                # print(second.results.multi_hand_landmarks)

                if second.results.multi_hand_landmarks:
                    for handLms in second.results.multi_hand_landmarks:
                        
                        for id, lm in enumerate(handLms.landmark):
                                            
                            if id ==8:
                                second.arr[0,0,0]=lm.x
                                second.arr[0,0,1]=lm.y
                                second.arr[0,0,2]=lm.z
                            elif id ==7:
                                second.arr[0,1,0]=lm.x
                                second.arr[0,1,1]=lm.y
                                second.arr[0,1,2]=lm.z
                            elif id ==6:
                                second.arr[0,2,0]=lm.x
                                second.arr[0,2,1]=lm.y
                                second.arr[0,2,2]=lm.z
                            elif id ==5:
                                second.arr[0,3,0]=lm.x
                                second.arr[0,3,1]=lm.y
                                second.arr[0,3,2]=lm.z
                            
                            elif id ==12:
                                second.arr[1,0,0]=lm.x
                                second.arr[1,0,1]=lm.y
                                second.arr[1,0,2]=lm.z
                            elif id ==11:
                                second.arr[1,1,0]=lm.x
                                second.arr[1,1,1]=lm.y
                                second.arr[1,1,2]=lm.z
                            elif id ==10:
                                second.arr[1,2,0]=lm.x
                                second.arr[1,2,1]=lm.y
                                second.arr[1,2,2]=lm.z
                            elif id ==9:
                                second.arr[1,3,0]=lm.x
                                second.arr[1,3,1]=lm.y
                                second.arr[1,3,2]=lm.z
                            
                            elif id ==16:
                                second.arr[2,0,0]=lm.x
                                second.arr[2,0,1]=lm.y
                                second.arr[2,0,2]=lm.z
                            elif id ==15:
                                second.arr[2,1,0]=lm.x
                                second.arr[2,1,1]=lm.y
                                second.arr[2,1,2]=lm.z
                            elif id ==14:
                                second.arr[2,2,0]=lm.x
                                second.arr[2,2,1]=lm.y
                                second.arr[2,2,2]=lm.z
                            elif id ==13:
                                second.arr[2,3,0]=lm.x
                                second.arr[2,3,1]=lm.y
                                second.arr[2,3,2]=lm.z
                            
                            elif id ==20:
                                second.arr[3,0,0]=lm.x
                                second.arr[3,0,1]=lm.y
                                second.arr[3,0,2]=lm.z
                            elif id ==19:
                                second.arr[3,1,0]=lm.x
                                second.arr[3,1,1]=lm.y
                                second.arr[3,1,2]=lm.z
                            elif id ==18:
                                second.arr[3,2,0]=lm.x
                                second.arr[3,2,1]=lm.y
                                second.arr[3,2,2]=lm.z
                            elif id ==17:
                                second.arr[3,3,0]=lm.x
                                second.arr[3,3,1]=lm.y
                                second.arr[3,3,2]=lm.z
                            
                            elif id ==4:
                                second.arr[4,0,0]=lm.x
                                second.arr[4,0,1]=lm.y
                                second.arr[4,0,2]=lm.z
                            elif id ==3:
                                second.arr[4,1,0]=lm.x
                                second.arr[4,1,1]=lm.y
                                second.arr[4,1,2]=lm.z
                            elif id ==2:
                                second.arr[4,2,0]=lm.x
                                second.arr[4,2,1]=lm.y
                                second.arr[4,2,2]=lm.z
                            elif id ==1:
                                second.arr[4,3,0]=lm.x
                                second.arr[4,3,1]=lm.y
                                second.arr[4,3,2]=lm.z
                
                            elif id ==0:
                                second.base[0,0]=lm.x
                                second.base[0,1]=lm.y
                                second.base[0,2]=lm.z

                            h, w, c = img.shape
                            cx, cy = int(lm.x *w), int(lm.y*h)
                            cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                        for idx, stri in enumerate(second.results.multi_handedness):
                            second.score=np.round((MessageToDict(stri).get('classification')[0])['score'] ,2)

                        for i in range(5):

                            vec1=second.arr[i,3]-second.base #5-0
                            vec2=second.arr[i,2]-second.arr[i,3] #6-5
                            vec3=second.arr[i,1]-second.arr[i,2] #7-6
                            vec4=second.arr[i,0]-second.arr[i,1] #8-7
                            
                            second.theta[3*i,0]=np.arccos(np.dot(vec3,vec4)/(np.linalg.norm(vec3)*np.linalg.norm(vec4))) #DIP
                            second.theta[3*i+1,0]=np.arccos(np.dot(vec2,vec3)/(np.linalg.norm(vec2)*np.linalg.norm(vec3))) #PIP
                            second.theta[3*i+2,0]=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) #MCP
                        
                        vec_1=-(second.base-second.arr[0][3])
                        vec_2=-(second.base-second.arr[3][3])

                        vec_bend=np.cross(vec_1,vec_2)
                        vec_bend=vec_bend/np.linalg.norm(vec_bend)
                        second.bend=abs(vec_bend[0][2])

                        second.result=(np.round(second.arr,2))
                        second.angles=180-np.degrees(second.theta)
                        second.angles_t=np.transpose(np.round(second.angles,2))
                        
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                else:
                    second.result=None
                    second.angles=np.zeros((15,1))
                    second.angles_t=np.zeros((1,15))
                    second.score=0
                    second.bend=0

                # print(second.angles_t)
                # print(second.score)
                # print(second.bend)

                cv2.imshow("Logitech_2", img)
                if cv2.waitKey(1) & 0xFF==ord("q"):
                    break

# obj=second() 
# obj.start()