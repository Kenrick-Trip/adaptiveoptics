# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:23:07 2021

@author: ktrip
"""

# assignment 3

#from cameras.ueye_camera import uEyeCamera
#from pyueye import ueye
#from scipy import ndimage

import numpy as np
#import matplotlib.pyplot as plt

SH_Sensor_Index = 2
Camera_Index = 1

#def grabframes(nframes, cameraIndex=0):
#    with uEyeCamera(device_id=cameraIndex) as cam:
#        cam.set_colormode(ueye.IS_CM_MONO8)#IS_CM_MONO8)
#        w=1280
#        h=1024
#        cam.set_aoi(0,0, w, h)
#        
#        cam.alloc(buffer_count=10)
#        cam.set_exposure(0.5)
#        cam.capture_video(True)
#    
#        imgs = np.zeros((nframes,h,w),dtype=np.uint8)
#        acquired=0
#        # For some reason, the IDS cameras seem to be overexposed on the 
#        # first frames (ignoring exposure time?). 
#        # So best to discard some frames and then use the last one
#        while acquired<nframes:
#            frame = cam.grab_frame()
#            if frame is not None:
#               imgs[acquired]=frame
#               acquired+=1
#           
#   
#       cam.stop_video()
#   return imgs

# zoom in on PSF:
    
def zoomImage(img, w, h):
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int((1040-w)/2)+i,int((1220-h)/2)+k]
    return zoomImg

# total cost function:
    
def costFunc(m1, m2, m3, m4):
    f1 = 1
    f2 = 1
    f3 = 1
    f4 = 1
    return f1*m1 + f2*m2 + f3*m3 + f4*m4

# all image metric functions: 
    
def sharpness(f):
    return np.sum(f)

def standardDev(image):
    [X,Y] = image.shape

    mu_x = 0
    mu_y = 0
    sigma_x = 0
    sigma_y = 0

    if X == Y:
        for i in range(X):
            mu_x += (i-X/2)*np.sum(image[i,:])
            mu_y += (i-Y/2)*np.sum(image[:,i])

        mu_x = mu_x/(np.sum(image))
        mu_y = mu_y/(np.sum(image))
        
        for i in range(i):
            sigma_x += (i - X/2 - mu_x)**2*np.sum(image[i,:]**2)
            sigma_y += (i - Y/2 - mu_y)**2*np.sum(image[:,i]**2)
    
        sigma_x = np.sqrt(sigma_x/np.sum(image**2))
        sigma_y = np.sqrt(sigma_y/np.sum(image**2))  

    else:
        #separate standard dev for x   
        for i in range(X):
            mu_x += (i-X/2)*np.sum(image[i,:])
            
        mu_x = mu_x/(np.sum(image))

        for i in range(i):
            sigma_x += (i - X/2 - mu_x)**2*np.sum(image[i,:]**2)

        sigma_x = np.sqrt(sigma_x/np.sum(image**2))

        #separate standard dev for x
        for j in range(Y):
            mu_y += (j-Y/2)*np.sum(image[:,j])

        mu_y = mu_y/(np.sum(image))  

        for j in range(Y):
            sigma_y += (j - Y/2 - mu_y)**2*np.sum(image[:,j]**2)

        sigma_y = np.sqrt(sigma_y/np.sum(image**2))
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
    #from dm.okotech.dm import OkoDM
    #with OkoDM(dmtype=1) as dm:   
        # define constants
        n = 100 # iterations per method
        w = 150
        h = 30
        
        val = np.zeros(n) # store total cost value
        act = np.zeros((n,19)) # len(dm))) # store actuator values
        offset = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # find static abberation using trail and error
        
        for i in range(n):
            f = np.zeros((n,w,h))
            act[i][:] = np.random.uniform(-1,1,size=19) + offset # len(dm)
            
            # send signal to DM
            #dm.setActuators(act[i][:])
            #img=grabframes(5, Camera_Index)

            img = np.random.rand(1280,1024)
            
            f[i][:][:] = zoomImage(img,w,h) # img[-1]
            
            # metrics -> Iij is PSF(i,j), here doneted as f[i][j]
            m1 = sharpness(f[i][:][:]) # metric 1 - sharpness
            m2 = standardDev(f[i][:][:]) # metric 2 - standard deviation
            m3 = secondMoment(f[i][:][:]) # metric 3 - second moment
            m4 = edgeSharpness(f[i][:][:]) # metric 4 - edge sharpness
            
            val[i] = costFunc(m1, m2, m3, m4)
        
        opt = np.argmin(val)
        opt_act = act[opt][:]
            
                
                
        

            
            
            
