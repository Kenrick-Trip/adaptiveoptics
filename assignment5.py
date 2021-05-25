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

import os as os
path = 'C:\\Users\\loekv\\OneDrive\\Documenten\\Tu Delft\\4de jaar\\Control for High Resolution Imaging\\Adaptive optics\\Scriptjes' #use double \ between two directories
os.chdir(path)


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
    
     threshold = np.mean(im2)*3 #determine threshold in less arbitrary way
     #Find the local coordinates on the total matrix 
     coordinates = peak_local_max(ShackHartmann, min_distance=10, indices = True, threshold_abs =  threshold)

    grid_ref = np.zeros((ShackHartmann.shape[0],ShackHartmann.shape[]))

     return coordinates, grid_ref 

def get_slopes(reference, coordinates, radius):
    
    
    
    ref_size = reference.shape[0]
    crd_size = coordinates.shape[0]
    
    
    
    if ref_size != crd_size:
        raise Warning('number of reference points differs from number of coordinates')
        
    for i in range(ref_size):
        
        centroid = reference[i,:]
        
        x_near = find_nearest(coordinates)
        
    

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index    


    

    
#%% Test code to test reference grid
im = plt.imread('plot1.PNG')
im= im[10:220,50:300,0]
plt.imshow(im)

#im = img_as_float(im)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=9, mode='constant')

im2 = np.around(im, decimals = 3)
mid = np.mean(im2)*1.5

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance= 10, indices = True, threshold_abs = mid)
#coordinates = coordinates[coordinates < 150]





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

#%%

im2 = np.around(im, decimals = 3)
mid = np.mean(im2)