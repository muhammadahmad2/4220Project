
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

import argparse
import sys, getopt


# In[8]:


def detect(img, cascade):
    rects = cascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


# In[9]:


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# In[16]:


#parser = argparse.ArgumentParser()    
#parser.add_argument("input", help="the file name")    
#args = parser.parse_args()

video_src = './VIDEO.MPG'#args.input
#args = dict(args)
#cascade_fn = 'C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml' #args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
#nested_fn  = 'C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml' #args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

#cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cascade = cv2.CascadeClassifier('./lbpcascade_frontalface_improved.xml')
nested = cv2.CascadeClassifier('./haarcascade_eye.xml')
smile = cv2.CascadeClassifier('./haarcascade_smile.xml')



# In[17]:


webcam = cv2.VideoCapture(video_src)
filter_img = cv2.imread('./hello.png')
height, width, channels = filter_img.shape

print(height, width)


# In[19]:


#webcam = cv2.VideoCapture(0)

#img = cv2.imread('./test.JPG')

while True:
    ret, img = webcam.read()

    #img = cv2.flip(img, 1)


    #img = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
    
    #rows,cols, chnls = img.shape
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    #img  = cv2.warpAffine(img,M,(cols,rows))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #t = clock()
    rects = detect(gray, cascade)
    vis = img.copy()
    
    if(rects != []):
        #print(rects[0])
        scale = 20
        x1,y1,x2,y2 = rects[0]
        new_width = int((x2+scale)-(x1-scale))
        new_height = int(((y2+scale)-(y1-scale))/4)
        
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            subrects = detect(roi.copy(), smile)
            
            if(subrects != []):
                filter_img = cv2.imread('./hearts.png')
            else:
                filter_img = cv2.imread('./hello.png')
                
            height, width, channels = filter_img.shape

        filter_scaled = cv2.resize(filter_img,(new_width, new_height))
      
        
        #print(new_width,new_height)
        #print( x1,y1,x2,y2)
        
        xt = x1-scale
        yt = y1-scale
        for w in range(new_width-1):
            
            for h in range(new_height-1):
                avg = np.average(filter_scaled[h,w])
                if((avg < 255) & (avg > 1)):
                    vis[yt,xt] = filter_scaled[h,w]
                yt += 1
            yt = y1-20-scale
            xt += 1
    
    #draw_rects(vis, rects, (0, 255, 0))
    #for x1, y1, x2, y2 in rects:
    #    roi = gray[y1:y2, x1:x2]
    #    vis_roi = vis[y1:y2, x1:x2]
    #    subrects = detect(roi.copy(), nested)
    #    draw_rects(vis_roi, subrects, (255, 0, 0))
    #dt = clock() - t

    #draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
    cv2.imshow('facedetect', vis)

    if 0xFF & cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()

