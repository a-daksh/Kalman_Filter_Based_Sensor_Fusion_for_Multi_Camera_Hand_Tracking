import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
import pandas as pd
from google.protobuf.json_format import MessageToDict

class first(Thread):
        
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
            super(first,self).__init__()
            pass

        def run(self):
            cap = cv2.VideoCapture(0)

            mpHands = mp.solutions.hands
            hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mpDraw = mp.solutions.drawing_utils


            while True:
                success, img = cap.read()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                first.results = hands.process(imgRGB)
                # print(results.multi_hand_landmarks)

                if first.results.multi_hand_landmarks:
                    for handLms in first.results.multi_hand_landmarks:
                        
                        for id, lm in enumerate(handLms.landmark):
                                            
                            if id ==8:
                                first.arr[0,0,0]=lm.x
                                first.arr[0,0,1]=lm.y
                                first.arr[0,0,2]=lm.z
                            elif id ==7:
                                first.arr[0,1,0]=lm.x
                                first.arr[0,1,1]=lm.y
                                first.arr[0,1,2]=lm.z
                            elif id ==6:
                                first.arr[0,2,0]=lm.x
                                first.arr[0,2,1]=lm.y
                                first.arr[0,2,2]=lm.z
                            elif id ==5:
                                first.arr[0,3,0]=lm.x
                                first.arr[0,3,1]=lm.y
                                first.arr[0,3,2]=lm.z
                            
                            elif id ==12:
                                first.arr[1,0,0]=lm.x
                                first.arr[1,0,1]=lm.y
                                first.arr[1,0,2]=lm.z
                            elif id ==11:
                                first.arr[1,1,0]=lm.x
                                first.arr[1,1,1]=lm.y
                                first.arr[1,1,2]=lm.z
                            elif id ==10:
                                first.arr[1,2,0]=lm.x
                                first.arr[1,2,1]=lm.y
                                first.arr[1,2,2]=lm.z
                            elif id ==9:
                                first.arr[1,3,0]=lm.x
                                first.arr[1,3,1]=lm.y
                                first.arr[1,3,2]=lm.z
                            
                            elif id ==16:
                                first.arr[2,0,0]=lm.x
                                first.arr[2,0,1]=lm.y
                                first.arr[2,0,2]=lm.z
                            elif id ==15:
                                first.arr[2,1,0]=lm.x
                                first.arr[2,1,1]=lm.y
                                first.arr[2,1,2]=lm.z
                            elif id ==14:
                                first.arr[2,2,0]=lm.x
                                first.arr[2,2,1]=lm.y
                                first.arr[2,2,2]=lm.z
                            elif id ==13:
                                first.arr[2,3,0]=lm.x
                                first.arr[2,3,1]=lm.y
                                first.arr[2,3,2]=lm.z
                            
                            elif id ==20: 
                                first.arr[3,0,0]=lm.x
                                first.arr[3,0,1]=lm.y
                                first.arr[3,0,2]=lm.z
                            elif id ==19:
                                first.arr[3,1,0]=lm.x
                                first.arr[3,1,1]=lm.y
                                first.arr[3,1,2]=lm.z
                            elif id ==18:
                                first.arr[3,2,0]=lm.x
                                first.arr[3,2,1]=lm.y
                                first.arr[3,2,2]=lm.z
                            elif id ==17:
                                first.arr[3,3,0]=lm.x
                                first.arr[3,3,1]=lm.y
                                first.arr[3,3,2]=lm.z
                            
                            elif id ==4:
                                first.arr[4,0,0]=lm.x
                                first.arr[4,0,1]=lm.y
                                first.arr[4,0,2]=lm.z
                            elif id ==3:
                                first.arr[4,1,0]=lm.x
                                first.arr[4,1,1]=lm.y
                                first.arr[4,1,2]=lm.z
                            elif id ==2:
                                first.arr[4,2,0]=lm.x
                                first.arr[4,2,1]=lm.y
                                first.arr[4,2,2]=lm.z
                            elif id ==1:
                                first.arr[4,3,0]=lm.x
                                first.arr[4,3,1]=lm.y
                                first.arr[4,3,2]=lm.z
                
                            elif id ==0:
                                first.base[0,0]=lm.x
                                first.base[0,1]=lm.y
                                first.base[0,2]=lm.z

                            h, w, c = img.shape
                            cx, cy = int(lm.x *w), int(lm.y*h)
                            cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                        for idx, stri in enumerate(first.results.multi_handedness):
                            first.score=np.round((MessageToDict(stri).get('classification')[0])['score'] ,2)
                        
                        for i in range(5):

                            vec1=first.arr[i,3]-first.base #5-0
                            vec2=first.arr[i,2]-first.arr[i,3] #6-5
                            vec3=first.arr[i,1]-first.arr[i,2] #7-6
                            vec4=first.arr[i,0]-first.arr[i,1] #8-7
                            
                            first.theta[3*i,0]=np.arccos(np.dot(vec3,vec4)/(np.linalg.norm(vec3)*np.linalg.norm(vec4))) #DIP
                            first.theta[3*i+1,0]=np.arccos(np.dot(vec2,vec3)/(np.linalg.norm(vec2)*np.linalg.norm(vec3))) #PIP
                            first.theta[3*i+2,0]=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) #MCP

                        vec_1=-(first.base-first.arr[0][3])
                        vec_2=-(first.base-first.arr[3][3])

                        vec_bend=np.cross(vec_1,vec_2)
                        vec_bend=vec_bend/np.linalg.norm(vec_bend)
                        first.bend=abs(vec_bend[0][2])

                        first.result=(np.round(first.arr,2))
                        first.angles=180-np.degrees(first.theta)
                        first.angles_t=np.transpose(np.round(first.angles,2))
                        
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                else:
                    first.result=None
                    first.angles=np.zeros((15,1))
                    first.angles_t=np.zeros((1,15))
                    first.score=0
                    first.bend=0

                # print(first.angles_t)
                # print(first.score)
                # print(first.bend)

                cv2.imshow("lenovo_0", img)
                if cv2.waitKey(1) & 0xFF==ord("q"):
                    break

# obj=first() 
# obj.start()