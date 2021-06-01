# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:24:33 2021

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
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from numpy.linalg import inv

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

def get_slopes(reference,grid_coor, coordinates, radius):
      
    ref_size = reference.shape[0]
    crd_size = coordinates.shape[0]

    if ref_size != crd_size:
        raise Warning('number of reference points differs from number of coordinates')
    

    difference = np.zeros((ref_size,6))    # [x0, y0, xref, yref delta_x, delta_y]
    for i in range(ref_size):
        
        centroid = reference[i,:] 
        
        for xr in range(-radius,radius):
            for yr in range(-radius,radius):
                x0 = centroid[0] + xr
                y0 = centroid[1] + yr
                
                if grid_coor[x0,y0] == 1:
                    difference[i,0] = x0
                    difference[i,1] = y0
                    difference[i,2] = centroid[0]
                    difference[i,3] = centroid[1]
                    difference[i,4] = x0 - centroid[0]
                    difference[i,5] = y0 - centroid[1]
                    break
            if grid_coor[x0,y0] == 1:
                break
            
            
    return difference

def distance(pos, threshold):
    
    """
    Calculates relative positions and distances between particles.
    """
    
    # print(pos.shape)
    
    num_spots = pos.shape[0]
    numDim = pos.shape[1]
    
    rel_pos = np.zeros((num_spots,num_spots,numDim))                                    
    rel_dist = np.zeros((num_spots,num_spots))                                           
    
    #calculate relative positions and distances
    rel_pos = np.subtract(pos,pos[:,None])
       
    rel_dist += np.sum(rel_pos**2, axis = 2)
    rel_dist = np.sqrt(rel_dist)
    
    indices = np.diag_indices(num_spots)                                    
    rel_dist[indices] = np.inf
    
    indices = np.where(rel_dist < threshold)
    
    indices = np.sort(indices,axis = 0)
    indices = np.unique(indices, axis = 0)
    
    pos = np.delete(pos, indices[0,:],axis = 0)
    
    # print(pos.shape)
    return pos

def corr_act_slope(R, A, slope):
    A = A.transpose
    n = slope.shape[1]
    print(n)
    S = np.zeros((1,2*n))
    
    for i in range(n):
        S[i,i] = slope[i,5]
        S[1,i+1] = slope[i,6]
        
    R = (R + S*(A.transpose)*inv(A*(A.transpose)))/2
    return R


if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:   
            
        # test with real image:
        n = 1000
        R = 0
        
        for i in range(n):
            A = np.random.uniform(-1,1,size=len(dm))
            dm.setActuators(A)    
            im = grabframes(3, 2)[-1] 
        
            # Comparison between image_max and im to find the coordinates of local maxima
            coordinates = peak_local_max(im, min_distance = 45, indices = True, threshold_abs = 3.5, num_peaks_per_label = 1)
            coordinates = distance(coordinates, 20)
            
            reference = 0 # get ref from assignment 6
            grid_coor = 0 # get grid coordinates from ?
            radius = 0 # what determines the radius ?
            slopes = get_slopes(reference, grid_coor, coordinates, radius)
            R = corr_act_slope(R, A, slopes)
            print(R)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        