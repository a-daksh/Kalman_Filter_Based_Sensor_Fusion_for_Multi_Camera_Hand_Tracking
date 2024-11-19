from threaded_Camera import *
from threaded_Camera_2 import *
import pandas as pd
import time
import csv

obj1=first()
obj2=second()

obj1.start()
obj2.start() 

X=180*np.ones((15,1))
sig=np.eye(15)
w=np.ones((15,1))
A= np.eye(15)
B= np.eye(15)  
C= np.vstack((np.eye(15), np.eye(15)))  
R=0.8*np.eye(15)

print("Loading....Please Wait")
time.sleep(20)
i=0

with open ("cam_mixed.csv","a",newline='',) as log:

    writer = csv.writer(log)
    writer.writerow(['Serial No.', 'Camera1', 'Camera2', 'Fused'])  

    while True:

        if obj1.score==0 and obj2.score==0:
            continue

        i+=1
        
        if obj1.score==0 and obj2.score!=0:
            Q1=100*np.eye(15)
            Q2=np.eye(15)/(obj2.score*(obj2.bend+0.1))

        elif obj1.score!=0 and obj2.score==0:
            Q1=np.eye(15)/(obj1.score*(obj1.bend+0.1))
            Q2=100*np.eye(15)
        
        else:
            Q1=np.eye(15)/(obj1.score*(obj1.bend+0.1))
            Q2=np.eye(15)/(obj2.score*(obj2.bend+0.1))

        Q=np.vstack((np.hstack((Q1,np.zeros((15,15)))),np.hstack((np.zeros((15,15)),Q2))))

        score1=obj1.score
        score2=obj2.score
        angles1=obj1.angles
        angles2=obj2.angles

        X=np.dot(A,X)+np.dot(0.1*B,w)
        X_prev=X
        sig=np.dot(A,np.dot(sig,np.transpose(A)))+R

        temp_1= np.dot(C,np.dot(sig,np.transpose(C)))+Q
        K=np.dot(np.dot(sig,np.transpose(C)),np.linalg.pinv(temp_1))
        Z=np.vstack((angles1,angles2))
        temp_2=Z-np.matmul(C,X)

        X=X+np.dot(K,temp_2)
        temp_3=np.eye(15)-np.dot(K,C)
        sig=np.dot(temp_3,sig)

        w=(X-X_prev)/0.11
        w=(w-np.min(w))/(np.max(w)-np.min(w))

        angles_t=np.transpose(np.round(X,2))
        angles_t_clip=np.clip(angles_t,0,180)

        arr=np.array([i,obj1.angles_t[0][0],obj2.angles_t[0][0],angles_t_clip[0][0]])
        print(arr.reshape(1,4))
        pd.DataFrame(arr.reshape(1,4)).to_csv(log,header=False,index=False)
        time.sleep(0.1)
