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

def add_disturbance(im, phase_screen):
    if np.max(phase_screen) > 0.1:
        return phase_screen.add_row()
    else:
        nx = im.shape[0]
        print(nx)
        r = 1280/1024
        lx = r/nx
        r0 = 0.1
        l0 = 100
        return PhaseScreenKolmogorov(nx,lx,r0,l0)
    
def add_noise(slopes):
    n = slopes.shape[0]
    dist = 0.01 # max size of disturbance
    for i in range(n):
        slopes[i,1] = slopes[i,1] + (0.5-np.rand.random(1,1))*2*dist
        slopes[i,2] = slopes[i,2] + (0.5-np.rand.random(1,1))*2*dist
    return slopes

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


def zernike_to_slopes(B, z):
    return np.dot(B, z)

def corr_act_slope(R, A, slope):
    A = A.transpose
    n = slope.shape[1]
    S = np.zeros((1,2*n))
    
    for i in range(n):
        S[1,i] = slope[i,1]
        S[1,i+1] = slope[i,2]
        
    R = (R + S*(A.transpose)*inv(A*(A.transpose)))/2
    return R, S

def act_from_slopes(R, slope):
    n = slope.shape[1]
    S = np.zeros((1,2*n))
    
    for i in range(n):
        S[1,i] = slope[i,1]
        S[1,i+1] = slope[i,2]
        
    return np.linalg.pinv(R)*S

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

def converge_to_zernike(dm, target_zernike):
    B = np.zeros(target_zernike) # (placeholder)
    
    # target slopes and 
    tar_s = zernike_to_slopes(B, target_zernike)

    #### control loop settings: ####
    
    # find some allowed error value
    err = 1e-3
    
    # constant gain
    Kp = 2 
    
    #### set all initial conditions: ####
    
    # find initial conditions actuator
    init_act = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, 
                -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 
                0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
    
    dm.setActuators(init_act)    
    
    # find initial image
    im = grabframes(3, 2)[-1] 
    
    # find initial slopes
    coordinates = peak_local_max(im, min_distance = 45, indices = True, threshold_abs = 3.5, num_peaks_per_label = 1)
    coordinates = distance(coordinates, 20)
    reference = 0 # get ref from assignment 6
    grid_coor = 0 # get grid coordinates from ?
    radius = 0 # what determines the radius ?
    s = get_slopes(reference, grid_coor, coordinates, radius)
    
    
    # find some allowed error value
    err = 1e-3
    
    # find correlation slopes and actuators:
    R_init = np.zeros((2*s.shape[1],len(dm)))
    R = corr_act_slope(R_init, init_act, (s - tar_s)*Kp)
    
    #https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-26-2-1655&id=380836
    
    
    #### control loop: ####
    
    while s - tar_s > err:
        err_s = (s - tar_s)*Kp # B matrix??
        act = act_from_slopes(R, err_s)
        
        
        dm.setActuators(act)    
        reference = 0 # get ref from assignment 6
        grid_coor = 0 # get grid coordinates from ?
        radius = 0 # what determines the radius ?
        s = get_slopes(reference, grid_coor, coordinates, radius)
        s = get_slopes() # (placeholder)
        R = corr_act_slope(R, act, (s - tar_s)*Kp)
        print(R)
    
    
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm: 
        # define zernike polynomial we want to converge to
        target_zernike = 60
        converge_to_zernike(dm, target_zernike)
