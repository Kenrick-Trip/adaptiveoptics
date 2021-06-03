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
# from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
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

def create_ref_grid(ShackHartmann):
     
    SH_round = np.around(ShackHartmann, decimals = 3)
    threshold = np.mean(SH_round)*1.5 #determine threshold in less arbitrary way
    #Find the local coordinates on the total matrix 
    coordinates = peak_local_max(ShackHartmann, min_distance= 45, indices = True, threshold_abs = 3.5)
    centers = distance(coordinates,10)
    
    grid_ref = np.zeros((ShackHartmann.shape[0],ShackHartmann.shape[1]))
    grid_ref[coordinates[:,0],coordinates[:,1]] = 1
    
    return centers, grid_ref

#def add_disturbance(im, phase_screen):
#    if np.max(phase_screen) > 0.1:
#        return phase_screen.add_row()
#    else:
#        nx = im.shape[0]
#        print(nx)
#        r = 1280/1024
#        lx = r/nx
#        r0 = 0.1
#        l0 = 100
#        return PhaseScreenKolmogorov(nx,lx,r0,l0)
    
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

    #if ref_size != crd_size:
    #    raise Warning('number of reference points differs from number of coordinates')
    

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


def zernike_to_slopes(B, z,im_unit):
    target_slopes = np.dot(B, z)
    target_slopes = target_slopes/(2/len(im_unit))
    return target_slopes

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
    # n = int(points/2)
    # slope = slope[0:n, :]
    # S = reshape_slopes(slope, points)
    
    res = A.dot(slope)
    # print(res.shape)
    return res.T

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

def converge_to_zernike(dm, target_zernike, points, A, iterations):
    #### TODO: find target slopes from Zernike ####
    # B = np.zeros(target_zernike) # (placeholder)
    # target slopes
    #tar_s = zernike_to_slopes(B, target_zernike)
    
    # test slopes:
    tar_s = np.zeros(points) #np.random.randint(5,size=(points)) - np.ones(points)*2    
    tar_act = act_from_slopes(A, tar_s, points)
    print(tar_act)
    
    #### control loop settings: ####   
    # find some allowed error value
    err = 1e-3
    # constant gain
    Kp = 0.8
    
    #### find reference image: ####
    ref_coordinates = create_ref_coordinates()
    
    #### set all initial conditions: ####
    init_act = np.random.uniform(-1,1,size=len(dm))  # np.zeros(len(dm)) #np.random.uniform(-1,1,size=len(dm)) 
    init_coordinates, init_grid = initial_conditions(init_act) 
    s = get_slopes(ref_coordinates, init_grid, init_coordinates, 6)
    n = int(points/2)
    s = s[0:n,:]
    s = reshape_slopes(s, points)
    

    act = init_act
    new_act = act_from_slopes(A, s, points)
    #new_act = np.transpose(new_act)
    new_act = new_act[0]
    print(new_act)
    
    #https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-26-2-1655&id=380836
    
    #iterations = 100
    
    error = np.zeros((iterations, len(dm)))
    error[0,:] = np.subtract(tar_act, new_act)
    # print(error[0,:])
    print(np.sum(np.abs(np.subtract(tar_act, new_act))))
    
    #### control loop: ####
    
    # while np.amax(error) > err:
    for i in range(iterations - 1):
        act = np.subtract(act, np.subtract(tar_act, new_act)).dot(Kp)
        act = np.clip(act, -1, 1)
        print(act)

        dm.setActuators(act)
        
        im2 = grabframes(3, 2)[-1] 
        coordinates, grid = create_ref_grid(im2)
        s = get_slopes(ref_coordinates, grid, coordinates, 6)
        n = int(points/2)
        s = s[0:n,:]
        s = reshape_slopes(s, points)
        
        new_act = act_from_slopes(A, s, points)
        #new_act = np.transpose(new_act)
        new_act = new_act[0]
        
        print(new_act)
        
        error[i+1,:] = np.subtract(tar_act, new_act)
        # print(error[i+1,:])
        print(np.sum(np.abs(np.subtract(tar_act, new_act))))
        
        # if np.all((new_act == 0)):
        #    break
    
    return tar_act, act, new_act, error, s

    
    
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm: 
        # define zernike polynomial we want to converge to
        A = np.load('influence_matrix.npy')
        target_zernike = 60
        points = 100
        iterations = 80
        tar_act, act, new_act, error, s = converge_to_zernike(dm, target_zernike, points, A, iterations)
        
        err = np.zeros(iterations)
        
        for i in range(iterations):
            err[i] = np.sum(np.abs(error[i,:]))
            
        it = np.arange(iterations)
        plt.plot(it, err)
