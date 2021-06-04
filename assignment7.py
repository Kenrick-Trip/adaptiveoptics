# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:55:45 2021

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
from numpy.linalg import inv
from scipy.optimize import least_squares

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

def create_ref_grid(ShackHartmann):
     
    SH_round = np.around(ShackHartmann, decimals = 3)
    threshold = np.mean(SH_round)*1.5 #determine threshold in less arbitrary way
    #Find the local coordinates on the total matrix 
    coordinates = peak_local_max(ShackHartmann, min_distance= 45, indices = True, threshold_abs = 3.5)
    centers = distance(coordinates,10)
    
    grid_ref = np.zeros((ShackHartmann.shape[0],ShackHartmann.shape[1]))
    grid_ref[coordinates[:,0],coordinates[:,1]] = 1
    
    return centers, grid_ref

def get_slopes(reference,grid_coor, coordinates, radius):
      
    ref_size = reference.shape[0]
    crd_size = coordinates.shape[0]
    
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

def reshape_slopes(slope, points):
    n = int(points/2)
    S = [(slope[:,4], slope[:,5].T)]
    S = np.reshape(S, (2*n,1), order='F')
    return S

def act_from_slopes(A, slope, points):
    res = A.dot(slope)
    # print(res.shape)
    return res

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

def create_ref_coordinates():
    # find initial conditions actuator
    opt_act = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, 
                -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 
                0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
    
    dm.setActuators(opt_act)    
    
    # find reference image
    im_ref = grabframes(3, 2)[-1] 
    coordinates,___ = create_ref_grid(im_ref)
    
    return coordinates

def initial_conditions(init_act):
    dm.setActuators(init_act)
    im2 = grabframes(3, 2)[-1] 
    
    # find initial slopes
    coordinates, grid = create_ref_grid(im2)
    
    return coordinates, grid

def kalman(A, C, y, R, Pprev, x_hat_prev):
    Kk = Pprev*(C.T)*(C*Pprev*(C.T)+R)**(-1)
    obs_gain = Kk.dot(C)
    dim = obs_gain.shape
    Lk = np.ones((dim)) - obs_gain
    x_hat = Kk.dot(y) + Lk.dot(x_hat_prev)
    Pk = Pprev - (Kk.dot(C)).dot(Pprev)
    return x_hat, Pk

def minimization(x_hat, N, act):
    #print(N.shape)
    #print(act.shape)
    f = (N.T).dot(act)
    
    return x_hat - f

def converge_to_ref(dm, points, iterations, N, A, C, R, P0, x0, x_hat0):
    # converge to reference
    tar_s = np.zeros(points)
    tar_act = act_from_slopes(N, tar_s, points)
    # print(tar_act)


    #### find reference image: ####
    ref_coordinates = create_ref_coordinates()
    
    #### set all initial conditions: ####
    init_act = x0
    init_coordinates, init_grid = initial_conditions(init_act) 
    s = get_slopes(ref_coordinates, init_grid, init_coordinates, 6)
    n = int(points/2)
    s = s[0:n,:]
    s = reshape_slopes(s, points)
    
    y = s
    # print(y.shape)
    
    act = init_act
    new_act = act_from_slopes(N, s, points)
    new_act = np.transpose(new_act)
    
    x_hat, Pk = kalman(A, C, y, R, P0, x_hat0)
    
     #### find error: ####
    error = np.zeros((iterations, len(dm)))
    error[0,:] = tar_act - new_act
    # print(error[0,:])
    print(np.sum(np.abs(tar_act - new_act)))
    
    #### control loop: ####
    
    # while np.amax(error) > err:
    for i in range(iterations - 1):
        #act = act.T
        #print(act.shape)
        #print(N.shape)
        
        for i in range(len(dm)):
            minim = least_squares(minimization, act[i], bounds=(-1, 1), args=(N[i,:], x_hat))
            act[i] = minim.x
            
        #act = act.T
        #print(act)

        dm.setActuators(act)
        
        im2 = grabframes(3, 2)[-1] 
        coordinates, grid = create_ref_grid(im2)
        s = get_slopes(ref_coordinates, grid, coordinates, 6)
        n = int(points/2)
        s = s[0:n,:]
        s = reshape_slopes(s, points)
        
        y = s
        
        new_act = act_from_slopes(N, s, points)
        new_act = np.transpose(new_act)
        
        # print(y)
        
        x_hat, Pk = kalman(A, C, y, R, Pk, x_hat)
        
        error[i+1,:] = tar_act - new_act
        
        # print(error[i+1,:])
        print(np.sum(np.abs(tar_s - s)))
        #print(minim.cost)
        
        #if minim.cost < 1e-2:
        #  break
    
    return tar_s, tar_act, act, y, error
    
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:   
        # set some initial conditions:
        P0 = 1
        x0 = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, 
                -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 
                0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
        x_hat0 = np.zeros((100,1))
        
                # settings
        N = np.load('influence_matrix.npy')
        points = 100
        iterations = 50
        
        # tunable constants
        R = 0.5
        A = np.ones((points, points))
        C = np.ones((points, points))
        
        # running the code
        tar_s, tar_act, act, y, error = converge_to_ref(dm, points, iterations, N, A, C, R, P0, x0, x_hat0)
        
        