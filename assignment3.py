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
        
            m1 = sharpness(f)
            m2 = 0 # metric 2
            m3 = 0 # metric 3
            m4 = 0 # metric 4
            
            val[i] = costFunc(m1, m2, m3, m4)
        
        opt = np.argmin(val)
        opt_act = act[opt][:]
            
                
                
        

            
            
            
