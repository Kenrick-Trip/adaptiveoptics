# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:01:59 2021

@author: Kenrick Trip
"""

# assignment 4

from cameras.ueye_camera import uEyeCamera
from pyueye import ueye
from scipy import ndimage

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import copy

SH_Sensor_Index = 2
Camera_Index = 1
progress = 0
maxiter = 5000

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
 
# TODO: plots - Actuators (Array plot) + Images (before, after) - in function format

def zoomImage(img, w, h):
    [wo, ho] = np.unravel_index(img.argmax(), img.shape)
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int(wo-w/2)+i,int(ho-h/2)+k]
    return zoomImg

def secondmoment(x):
    print(x)
        
    dm.setActuators(x)
    img = grabframes(3, Camera_Index)
    
    f = img[-1]
    
    w = 40
    h = 40
    image = zoomImage(f, w, h)
    
    [X,Y] = image.shape
    
    mx = my = 0
    if X == Y:
        for i in range(X):
            mx += (i-w/2)**2*np.sum(image[i,:])
            my += (i-h/2)**2*np.sum(image[:,i])
    else:
        for i in range(X):
            mx += (i-w/2)**2*np.sum(image[i,:])
        for i in range(Y): 
            my += (i-h/2)**2*np.sum(image[:,i])
    
    global progress
    progress = progress + 1
    percentage = round((progress/maxiter)*100,2)
    print('progress = {}%'.format(percentage))
    
    res = (mx+my)*10**(-5)
    print(res)
    
    return res 

def secondmoment1(x):
    print(x)
        
    dm.setActuators(x)
    img = grabframes(3, Camera_Index)
    
    f = img[-1]
    [wo, ho] = np.unravel_index(f.argmax(), f.shape)
    
    w = 50
    h = 50
    image = zoomImage(f, w, h)
    
    [X,Y] = image.shape
    
    M = 0
    if X == Y:
        for i in range(X):
            for k in range(Y):
                M += (i-25)*(k-25)*image[i,k]
    
    global progress
    progress = progress + 1
    percentage = round((progress/maxiter)*100,2)
    print('progress = {}%'.format(percentage))
    
    print(M)
    
    return M
    
    
def standardDev(x):
    print(x)
        
    dm.setActuators(x)
    img = grabframes(3, Camera_Index)

    w = 30
    h = 30
    image = zoomImage(img[-1], w, h)
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
    
    global progress
    progress = progress + 1
    percentage = round((progress/maxiter)*100,2)
    print('progress = {}%'.format(percentage))
    
    res = sigma_x+sigma_y
    print(res)
    
    return res


if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:           
        # send signal to DM
        A = np.zeros(len(dm))
        
        # inner circle
        A[0] = 0
        A[1] = 1
        A[2] = -0.5
        A[3] = 0
        A[4] = 1           
        
        # outer circle
        A[5] = 0.5
        A[6] = 0.5
        A[7] = 0.5
        A[9] = 0.5
        A[10] = 0.5
        
        
        
        # tip/tilt
        A[17] = 1
        A[18] = -0.667
        
        B = np.zeros(len(dm))
        C = np.zeros((len(dm)+1, len(dm)))
        for i in range(len(dm)):
            C[i,:] = (np.random.randint(21, size=19)-10*np.ones(19))/10
        
        Aopt = minimize(secondmoment, B, method='nelder-mead', 
                        options={'initial_simplex': C, 'xatol': 1e-6, 'disp': True, 'maxfev': maxiter, 'fatol': 1, 'adaptive': False})