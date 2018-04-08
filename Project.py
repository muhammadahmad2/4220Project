
import os
import cv2
import sys
import random
import numpy as np

class Eigenfaces(object):                                                       
    l = 400                                      
    m = 92                                                                      
    n = 112                                                                     
    mn = m * n                                                                  
    
    def __init__(self, _energy = 0.85):
        self.energy = _energy
        
        L = np.empty(shape=(self.mn, self.l), dtype='float64')

        i = 0
        for imgfolder in os.listdir('./Bioid/'):
            for filename in os.listdir('./Bioid/' + imgfolder):
                filename = './Bioid/' + imgfolder + '/'+ filename
                if (filename.lower().endswith(('.png', '.jpg', '.jpeg','.pgm'))):
                    img=cv2.imread(filename,0)

                    img = cv2.resize(img, (92,112), interpolation = cv2.INTER_AREA)

                    img_col = np.array(img, dtype='float64').flatten()
                    L[:, i] = img_col[:]                              
                    i += 1                                           

        self.mean_img_col = np.sum(L, axis=1) / self.l            
        
        for j in range(0, self.l):                                
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)      
        C /= self.l                                                             

        self.evalues, self.evectors = np.linalg.eig(C)                          
        sort_indices = self.evalues.argsort()[::-1]                             
        self.evalues = self.evalues[sort_indices]                               
        self.evectors = self.evectors[sort_indices]                             
        
        evalues_sum = sum(self.evalues[:])                                      
        evalues_count = 0                                                       
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            
        self.evectors = self.evectors[0:evalues_count]

        self.evectors = self.evectors.transpose()                               
        self.evectors = L * self.evectors                                       
        norms = np.linalg.norm(self.evectors, axis=0)                          
        self.evectors = self.evectors / norms  
        self.W = self.evectors.transpose() * L
        
    def classify(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(gray)

        
        img = cv2.resize(img,(92, 112), interpolation = cv2.INTER_AREA)   
        img_col = np.array(img, dtype='float64').flatten()  
        img_col1=img_col
        img_col -= self.mean_img_col                                         
        img_col2 = img_col
        img_col = np.reshape(img_col, (self.mn, 1))                         
        img_col3 =img_col
        
        S = self.evectors.transpose() * img_col                                
        
        diff = self.W - S                                                       
        
        norms = np.linalg.norm(diff, axis=0)

        liv = np.argmin(norms)                                     
        return liv,norms
    
def draw_rects(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    
if __name__ == "__main__":
    efaces = Eigenfaces()
    
    cam = cv2.VideoCapture('./video.mpg')
    images = []
    runtime = 0
    height1, width1, channels1 = 0,0,0
    import time
    start = time.time()
    
    while True:
        ret,img = cam.read()
        runtime +=1
        
        if ret == False:
            break

        height1, width1, channels1 = img.shape
        
        div = int(width1/100)
    
        if(div == 0):
            div = 1

        vis = cv2.resize(img,(int(width1/div), int(height1/div)), interpolation = cv2.INTER_AREA)
        height, width, channels = vis.shape

        x = 92
        y = 112

        nom=[]
        xy = []
        for i in range(4):
            w = 0
            for a in range(width):
                h = 0
                if((w + x)>width):
                    break
                for b in range(height):
                    if((h + y)>height):
                        break
                    temp = vis[h : h + y, w : w + x]
                    liv,norms = efaces.classify(temp)
                    nom.append(norms[liv])
                    xy.append([h*div,y*div,w*div,x*div])
                    h += 5

                w += 5
            x -= 20
            y -= 24
                
        low = np.argmin(nom)
        h,y,w,x=xy[low]
        if(nom[low]<=3800):     
            draw_rects(img, w,h,w + x,h + y, (255,0,0))
        
        images.append(img)
        
    done = time.time()
    elapsed = done - start
    print(elapsed)
    
        
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width1,height1))

    for image in images:
        out.write(image)

    out.release()
