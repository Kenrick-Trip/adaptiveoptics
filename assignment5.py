# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:54:31 2021

@author: loekv
"""
import numpy as np
import scipy as scp
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from cameras.ueye_camera import uEyeCamera
from pyueye import ueye
from sklearn.cluster import KMeans




#%%
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
    # TODO: Load Shack hartmann image
     #ShackHartmann= []
     
     SH_round = np.around(ShackHartmann, decimals = 3)
     threshold = np.mean(SH_round)*3 #determine threshold in less arbitrary way
     #Find the local coordinates on the total matrix 
     coordinates = peak_local_max(ShackHartmann, min_distance=10, indices = True, threshold_abs =  threshold)

     #grid_ref = np.zeros((ShackHartmann.shape[0],ShackHartmann.shape[1]))
     #grid_ref[coordinates[:,0],coordinates[:,1]] = 1

    centers = distance(coordinates)
    
     return centers

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

def Clusters(coordinates):
    kmeans = KMeans(n_clusters=150)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
        
    
def distance(pos, threshold):
    """
    Calculates relative positions and distances between particles.
    """
    print(pos.shape)
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
    print(pos.shape)
    return pos
    
#%% Test code to test reference grid

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:   
            
        #im = plt.imread('plot1.PNG')
        #im= im[10:220,50:300,0]
        #plt.imshow(im)
        
        # test with real image:
        A =  np.random.uniform(-1,1,size=len(dm))
        B = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
        dm.setActuators(B)    
        im = grabframes(5, 2)[-1] 
        im = im[70:780,300:950]
            
        #im = img_as_float(im)
        
        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=45, mode='constant')
        
        im2 = np.around(im, decimals = 3)
        mid = np.mean(im2)*1.5
        
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance = 45, indices = True, threshold_abs = 3.5, num_peaks_per_label = 1)
        #coordinates = coordinates[coordinates < 150]
        
        kmeans = KMeans(n_clusters=150)
        kmeans.fit(coordinates)
        centers = kmeans.cluster_centers_
        
        
        
        # display results
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True )
        ax = axes.ravel()
        ax[0].imshow(im, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')
        
        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')
        
        ax[2].imshow(im, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')
        
        fig.tight_layout()
        
        plt.show()
        

        
        
        
        
