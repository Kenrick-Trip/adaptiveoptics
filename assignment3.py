# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:23:07 2021

@author: ktrip
"""

# assignment 3

from cameras.ueye_camera import uEyeCamera
from pyueye import ueye
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt

SH_Sensor_Index = 2
Camera_Index = 1
progress = 0

def grabframes(nframes, cameraIndex=0):
    with uEyeCamera(device_id=cameraIndex) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)#IS_CM_MONO8)
        w=1280
        h=1024
        cam.set_aoi(0,0, w, h)
        
        cam.alloc(buffer_count=10)
        cam.set_exposure(0.5)
        cam.capture_video(True)
    
        imgs = np.zeros((nframes,h,w),dtype=np.uint8)
        acquired=0
        # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?). 
        # So best to discard some frames and then use the last one
        while acquired<nframes:
            frame = cam.grab_frame()
            if frame is not None:
                imgs[acquired]=frame
                acquired+=1
            
    
        cam.stop_video()
    
    return imgs

# zoom in on PSF:
    
def zoomImage(img, w, h):
    wo = -296 #270
    ho = 83 #138
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int((1280+wo*2-w)/2)+i,int((1024+ho*2-h)/2)+k]
    return zoomImg

# total cost function:
    
def costFunc(m1, m2, m3, m4):
    f1 = 0
    f2 = 1
    f3 = 0
    f4 = 0
    return f1*m1 + f2*m2 + f3*m3 + f4*m4

# all image metric functions: 
    
def sharpness(f):
    return np.sum(f)

def standardDev(image):
    [X,Y] = image.shape

    X_vec = np.arange(0,X)-X/2
    Y_vec = np.arange(0,Y)-Y/2

    mu_x = X_vec*np.sum(image,axis = 1)
    mu_x = np.sum(mu_x)/np.sum(image)
    mu_y = Y_vec*np.sum(image,axis = 0)
    mu_y = np.sum(mu_y)/np.sum(image)

    sigma_x = (X_vec - mu_x)**2*np.sum(image**2, axis = 1)
    sigma_x = np.sqrt(np.sum(sigma_x)/np.sum(image**2))
    sigma_y = (Y_vec - mu_x)**2*np.sum(image**2, axis = 0)
    sigma_y = np.sqrt(np.sum(sigma_y)/np.sum(image**2))

    return sigma_x+sigma_y

def secondMoment(image):
    [X,Y] = image.shape
    
    mx = my = 0
    if X == Y:
        for i in range(X):
            mx += (i-X/2)**2*np.sum(image[i,:])
            my += (i-Y/2)**2*np.sum(image[:,i])
    else:
        for i in range(X):
            mx += (i-X/2)**2*np.sum(image[i,:])
        for i in range(Y): 
            my += (i-Y/2)**2*np.sum(image[:,i])
    return  mx+my

def edgeSharpness(f):
    [X,Y] = f.shape
    S1 = 0
    S2 = 0
    
    for i in range(X-1):
        for j in range(Y-1):
            S1 = S1 + (f[i+1,j]-f[i,j])**2 + (f[i,j+1]-f[i,j])**2
            S2 = S2 + f[i,j]
    
    return S1/S2

# TODO: insert metrics 
# TODO: plots - Actuators (Array plot) + Images (before, after) - in function format

# main loop:
    
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:   
        # define constants
        n = 4000 # iterations per method
        w = 20
        h = 20
        progress = 0
        
        val = np.zeros(n) # store total cost value
        act = np.zeros((n,19)) # len(dm))) # store actuator values
        offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # find static abberation using trail and error
        
        for i in range(n):
            f = np.zeros((n,h,w))
            #act[i][:] = np.random.uniform(-1,1,size=19) + offset # len(dm)
            #x = (np.random.randint(5, size=5)-2*np.ones(5))/2
            #x = (np.random.randint(7, size=5)-3*np.ones(5))/3 tip/tilt
            x = (np.random.randint(2, size=12))/2
            A = np.zeros(len(dm))
            
            A[0] = 0
            A[1] = 1
            A[2] = -0.5
            A[3] = 0
            A[4] = 1
            A[17] = 1
            A[18] = -0.667
            
            for k in range(12):
                A[k+5] = x[k]
            
            act[i][:] = A
            print(A)
            
            #global progress
            progress = progress + 1
            percentage = round((progress/n)*100,2)
            print('progress = {}%'.format(percentage))
            
            # send signal to DM
            dm.setActuators(act[i][:])
            img=grabframes(5, Camera_Index)

            #img = np.random.rand(1280,1024)
            
            f[i][:][:] = zoomImage(img[-1],h,w) # img[-1]
            
            # metrics -> Iij is PSF(i,j), here doneted as f[i][j]
            m1 = sharpness(f[i][:][:]) # metric 1 - sharpness
            m2 = standardDev(f[i][:][:]) # metric 2 - standard deviation
            m3 = secondMoment(f[i][:][:]) # metric 3 - second moment
            m4 = edgeSharpness(f[i][:][:]) # metric 4 - edge sharpness
            
            val[i] = costFunc(m1, m2, m3, m4)
        
        opt = np.argmin(val)
        opt_act = act[opt][:]
            
                
                
        

            
            
            
