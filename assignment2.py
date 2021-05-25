"""
Example for PC in right corner

Setup right corner:
    Camera 1: Regular camera
    Camera 2: SH sensor
"""

from cameras.ueye_camera import uEyeCamera
from pyueye import ueye
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt

SH_Sensor_Index = 2
Camera_Index = 1
  
def zoomImage(img, w, h):
    [wo, ho] = np.unravel_index(img.argmax(), img.shape)
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int(wo-w/2)+i,int(ho-h/2)+k]
    return zoomImg

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

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:
    
    
        print(f"Deformable mirror with {len(dm)} actuators")
        
        # send signal to DM
        n = 1# len(dm)
        
        for i in range(n):
            
            # send signal to DM
            A = np.zeros(len(dm))
            
            # inner circle
            A[0] = 0
            A[1] = 1
            A[2] = -0.5
            A[3] = 0
            A[4] = 1           
            
            # outer circle
            A[5] = 0.5
            A[6] = 0.5
            A[7] = 0.5
            A[9] = 0.5
            A[10] = 0.5
            
            
            
            # tip/tilt
            A[17] = 1
            A[18] = -0.667
            
            # nelder mead B=standardDev, C=secondmoment
            B = [0.0867, 0.0301, -0.6900, 0.0404, 0.5881, -0.1695, 0.1227, -0.3075, -0.2758, -0.3140, 0.4008, 0.6092, 0.0031, -0.5832, 0.4570, -0.7946, 0.3021, 0.0327, 0.3566]
            C = [-0.73711375, -0.86226782,  0.44749225, -0.18944923,  0.15651304, -0.86378585,
                 0.50516737,  0.74433535,  0.12245099,  0.20117577, -0.22835749,  0.40728353,
                 0.34554966, -0.00241688, -0.34318344,  0.28183828,  0.6185486,  -0.67263108,
                 0.62322176]
            
            # old setting: np.random.uniform(-1,1,size=len(dm))
            
            dm.setActuators(B)
            
            #plt.figure()    
            #img=grabframes(5, Camera_Index)
            #plt.imshow(img[-1])
            # take SH image
            
            plt.figure()
            img=grabframes(5, SH_Sensor_Index)
            #plt.imshow(img[-1], cmap='hsv')
            
            #plt.figure()
            #img=grabframes(3, Camera_Index)
        
            #fig1 = ndimage.zoom(img[-1], 0.25)
            
            #img1 = img[-1]
            
            #w = 30 #150
            #h = 30 #30
            
            #fig2=zoomImage(img[-1], w, h)
            
            SHarray = img[-1]
            SHmax = np.max(SHarray)
            
            SHpoints = np.zeros((1024,1280))
            
            for i in range(1280):
                for k in range(1024):
                    if SHarray[k,i] > 1.2:
                        SHpoints[k,i] = 1
            
            plt.scatter(SHpoints)
            #plt.imshow(SHpoints, cmap='gray')
            #plt.imshow(img[-1]) #,aspect=1/7) #, cmap='gist_ncar')
            #plt.colorbar()
            
            # untested code: saving image
            #directionary = "\\tudelft.net\student-homes\T\ktrip\Desktop\SC42065\Assignment2figs" # define some folder 
            #name = "actuator {}".format(i)  # create different image names
            #plt.savefig("{}/{}.png".format(directionary,name))
