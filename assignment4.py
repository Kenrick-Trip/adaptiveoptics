# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:01:59 2021

@author: Kenrick Trip
"""

# assignment 4

#from cameras.ueye_camera import uEyeCamera
#from pyueye import ueye
#from scipy import ndimage

import numpy as np
from scipy.optimize import minimize
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
#                imgs[acquired]=frame
#                acquired+=1
#           
#   
#        cam.stop_video()
#    return imgs
 
# TODO: plots - Actuators (Array plot) + Images (before, after) - in function format

def zoomImage(img, w, h):
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int((1280-w)/2)+i,int((1024-h)/2)+k]
    return zoomImg

def secondmoment(A):
    dm.setActuators(A)
    img = grabframes(5, Camera_Index)
    
    w = 400
    h = 400
    image = zoomImage(img[-1], w, h)
    
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


if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:           
        x0 = np.ones(19) # len(dm)
        # x0 = A ex3
        
        Aopt = minimize(secondmoment, x0, method='nelder-mead', 
                        options={'xatol': 1e-3, 'disp': True, 'maxiter': 4000})
            