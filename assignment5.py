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
import time as time
from zernike import RZern



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
     
    SH_round = np.around(ShackHartmann, decimals = 3)
    threshold = np.mean(SH_round)*1.5 #determine threshold in less arbitrary way
     #Find the local coordinates on the total matrix 
    coordinates = peak_local_max(ShackHartmann, min_distance= 10, indices = True, threshold_abs = threshold)
    centers = distance(coordinates,10)
    
    grid_ref = np.zeros((ShackHartmann.shape[0],ShackHartmann.shape[1]))
    grid_ref[coordinates[:,0],coordinates[:,1]] = 1
    
    return centers, grid_ref


def get_slopes(reference,grid_coor, coordinates, radius):
      
    ref_size = reference.shape[0]
    crd_size = coordinates.shape[0]

    if ref_size != crd_size:
        raise Warning('number of reference points differs from number of coordinates')
    

    difference = np.zeros((ref_size,6))    # [x0, y0, xref, yref delta_x, delta_y]
    for i in range(ref_size):
        
        centroid = reference[i,:] 
        n = 0
        for xr in range(-radius,radius):
            for yr in range(-radius,radius):
                x0 = centroid[0] + xr
                y0 = centroid[1] + yr
                
                
                if grid_coor[x0,y0] == 1:
                    difference[i,0] += x0 
                    difference[i,1] += y0
                    n+= 1
                    difference[i,2] = centroid[0]
                    difference[i,3] = centroid[1]

                    #break
            #if grid_coor[x0,y0] == 1:
             #   break
        difference[i,0] = np.florr(difference[i,0]/n)
        difference[i,1] = np.floor(difference[i,1]/n)
            
        difference[i,4] = difference[i,0] - centroid[0]
        difference[i,5] = difference[i,1] - centroid[1]
    return difference

def distance(pos, threshold):
    """
    Calculates relative positions and distances between particles.
    """
    #print(pos.shape)
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
    #print(pos.shape)
    return pos


def Zernike(mode,im_unit):
    cart = RZern(6)
    x = np.linspace(-1,1,len(im_unit))
    y = np.linspace(-1,1,len(im_unit))
    xv, yv = np.meshgrid(x,y)
    
    cart.make_cart_grid(xv, yv)
    c = np.zeros(cart.nk)
    c[mode] = 1.0
    Phi = cart.eval_grid(c, matrix=True)
    
    return Phi


def B_matrix(im,coordinates,slopes,modes):
    #Diameter of sensor in x and y direction
    Dx = np.max(coordinates[:,0])-np.min(coordinates[:,0])
    Dy = np.max(coordinates[:,1])-np.min(coordinates[:,1])
    D = np.maximum(Dx,Dy)

    #center of sensor
    midy = np.int(np.max(coordinates[:,1])-Dy/2)
    midx = np.int(np.max(coordinates[:,0])-Dx/2)

    #Maximale uitwijking tov midden 
    Ry = np.int(np.max(np.abs(coordinates[1]-midy)))
    Rx = np.int(np.max(np.abs(coordinates[0]-midx)))
    R = np.maximum(Rx,Ry)+30
    
    #Cropped image to fit unit circle
    im_unit = im[midx-R:midx+R,midy-R:midy+R]
  
    #transformed coordinates of centres
    cor_unit = coordinates - np.ones((len(coordinates),2))*[midx-R,midy-R]
    cor_unit = cor_unit.astype(int)
    slopes_unit = slopes*(2/len(im_unit))
   
    x = np.linspace(-1,1,len(im_unit))
    y = np.linspace(-1,1,len(im_unit))


    xv, yv = np.meshgrid(x,y)
    plt.figure()
    plt.pcolor(xv,yv,im_unit)
    plt.title('Unit grid')
    plt.show()
    
    ## Zernike
    cart = RZern(6)
    cart.make_cart_grid(xv, yv)
    c = np.zeros(cart.nk)
    B = np.zeros((coordinates.shape[0]*2,modes))

    

    
    for i in range(1, modes): #set desired zernike range
        
        c *= 0.0
        c[i] = 1.0
        Phi = cart.eval_grid(c, matrix=True)
        Zr = Phi[cor_unit[:,0],cor_unit[:,1]] #Zernike function at reference points
       
        Zx = Phi[cor_unit[:,0]+slopes[:,0],cor_unit[:,1]] #reference points plus delta y
        grady = (Zx-Zr)/slopes_unit[:,0]
        print(grady)
    
        Zy = Phi[cor_unit[:,0],cor_unit[:,1]+slopes[:,1]] #reference points plus delta x
        gradx = (Zy-Zr)/slopes_unit[:,1]
        print(gradx)
        
        B[:,i] = np.concatenate((gradx, grady))
        B[np.isnan(B)] =0

    return B, im_unit

def wavefront_reconstruction(B,slopes,modes,im_unit):
    slopes = np.reshape(slopes, 50, order='F')
    inverse = np.linalg.pinv(B).dot(slopes)
    zernike = np.zeros(im_unit.shape)
    

    for i in range(modes):
        zernike = zernike + inverse[i]*Zernike(i,im_unit)

    plt.imshow(zernike)
    return inverse 



    
#%% Test code to test reference grid

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:   
            
        #im = plt.imread('plot1.PNG')
        #im= im[10:220,50:300,0]
        
        
        # test with real image:
        A =  np.random.uniform(-1,1,size=len(dm))
        B = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
        dm.setActuators(B)    
        im = grabframes(3, 2)[-1] 

    
         
        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=45, mode='constant')
        
        im2 = np.around(im, decimals = 3)
        mid = np.mean(im2)*1.5
        
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance = 45, indices = True, threshold_abs = 3.5, num_peaks_per_label = 1)
        coordinates = distance(coordinates, 20)
        
        
        
        
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


slopes = np.random.randint(5, size = (len(coordinates),2))
modes = 10
B, im_unit = B_matrix(im,coordinates,slopes,modes)

B[np.isnan(B)] =0


inverse = wavefront_reconstruction(B,slopes,modes, im_unit)

        
        
#%% Test get_slopes
plt.figure()
im = plt.imread('plot1.PNG')
im= im[10:220,50:300,0]
plt.imshow(im)

plt.figure()
im2 = plt.imread('plot19.PNG')
im2= im2[10:220,50:300,0]
plt.imshow(im2)


#coordinates1 = peak_local_max(im, min_distance= 10, indices = True, threshold_abs = mid)
start = time.time()
coordinates1,___ = create_ref_grid(im)

#coordinates2 = peak_local_max(im2, min_distance= 10, indices = True, threshold_abs = mid)
coordinates2,grid2 = create_ref_grid(im2)


difference = get_slopes(coordinates,grid2, coordinates2,6)




stop = time.time()
print(stop - start)
plt.figure()
plt.plot(difference[:,0],difference[:,1], 'r.')
plt.xlim(0,250)
plt.ylim(0,220)

plt.show()


plt.figure()
plt.plot(coordinates1[:,0],coordinates1[:,1], 'r.')
plt.plot(difference[:,0],difference[:,1], 'b.')
plt.xlim(0,250)
plt.ylim(0,220)

plt.show()     
        
