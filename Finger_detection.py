import cv2
import numpy as np
import imutils
import math
#lo = np.array([0,130,101])
#hi = np.array([198,155,148])
cap = cv2.VideoCapture(0)

def putText(cv2,gray,txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray,'Fingers:'+txt,(0,70), font, 1,(255,0,0),2,cv2.LINE_AA)

while(1):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    kernel = np.ones((5,5),np.uint8)
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
    gray = cv2.medianBlur(gray,5)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    cont,hier = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cont, key = lambda x: cv2.contourArea(x))
    cov = len(cont)
    #numb = np.array([])
    if(cov>0):
        for i in range(cov):
            covt = cont[i]
            ar = cv2.contourArea(covt)
            if(ar>230):
                M = cv2.moments(covt)
                x1 = int(M['m10']/M['m00'])
                y1 = int(M['m01']/M['m00'])
                centroid = (x1,y1)
                #numb = np.append(numb,x1)
                cv2.circle(frame,(x1,y1),10,(0,24,206),-1)
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    hull = cv2.convexHull(cnt)
    cv2.drawContours(frame,[cnt],-1,(0,255,0),thickness=2)
    
    #cv2.drawContours(frame,[hull],-1,(0,255,0),2)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    #cv2.drawContours(frame,cont,-1,(255,0,0),3)
    count_defects = 0
    pt=[]
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        
        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])  **2 + (end[1] - far[1])  **2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(frame, far, 5, [0,0,255], -1)
            pt.append((far[0],far[1]))
        
        #elif len(pt)>3:
            #cv2.line(frame,pt[0],pt[1],[0,255,0],2)
            #draw = cnt[pt[0]:pt[1]]
            #print(draw)    
            #print('pt[0]',pt[0],'pt[1]',pt[1])
            #cnt=list(cnt)
            #print(cnt.index(pt[0]))
                        
            #cv2.line(frame,pt[1],pt[2],[255,0,0],2)
            #cv2.line(frame,pt[2],pt[3],[255,0,0],2)
            #cv2.line(frame,pt[3],pt[4],[255,0,0],2)
            #cv2.line(frame,pt[4],pt[0],[255,0,0],2)
            #cv2.line(frame,start,pt[0],[255,0,0],2)
        cv2.circle(frame, start, 5, [0,0,255], -1)
          
    putText(cv2,frame,str(count_defects+1))
    cv2.imshow('Frame',frame)
    cv2.imshow('Thresh',thresh)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('hand1.jpg',frame)

    if cv2.waitKey(27) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
