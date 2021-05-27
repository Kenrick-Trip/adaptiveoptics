# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:15:07 2021

@author: Kenrick Trip
"""

import numpy as np
import scipy as scp
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from cameras.ueye_camera import uEyeCamera
from pyueye import ueye

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

# TODO finish slopes to act!
def slopes_to_act(s):
    A = np.zeros(19) # (placeholder)
    return A

def zernike_to_slopes(B, z):
    return np.dot(B, z)

def converge_to_zernike(dm):
    # all target settings:
    tar_act = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, 
               -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 
               0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
    
    # TODO implement Q5 to find target zernike!
    
    # target Zernike tar_z (placeholder)
    n = 10
    tar_z = np.zeros(n) # (placeholder)
    B = np.zeros(n) # (placeholder)
    
    # target slopes and 
    tar_s = zernike_to_slopes(B, tar_z)
    Kp = 2 
    
    # find current slopes
    s = get_slopes() # (placeholder)
    
    # find some allowed error value
    err = 1e-3
    
    while s - tar_s > err:
        act = (s - tar_s)*Kp
        s = get_slopes() # (placeholder)
    
    
    
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm: 
        converge_to_zernike(dm)
