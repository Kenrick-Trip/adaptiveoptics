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
    wo = -294 #270
    ho = 84 #138
    zoomImg = np.zeros((w,h))
    for i in range(w):
        for k in range(h):
             zoomImg[i,k] = img[int((1280+wo*2-w)/2)+i,int((1024+ho*2-h)/2)+k]
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
            
            
            # old setting: np.random.uniform(-1,1,size=len(dm))
            
            dm.setActuators(A)
            
            #plt.figure()    
            #img=grabframes(5, Camera_Index)
            #plt.imshow(img[-1])
            # take SH image
            
            #plt.figure()
            #img=grabframes(5, SH_Sensor_Index)
            #plt.imshow(img[-1], cmap='gray')
            
            plt.figure()
            img=grabframes(2, Camera_Index)
        
            fig1 = ndimage.zoom(img[-1], 0.25)
            
            img1 = img[-1]
            
            w = 20 #150
            h = 20 #30
            
            fig2=zoomImage(img[-1], w, h)
            
          
            
            plt.imshow(fig2) #,aspect=1/7) #, cmap='gist_ncar')
            plt.colorbar()
            
            # untested code: saving image
            #directionary = "\\tudelft.net\student-homes\T\ktrip\Desktop\SC42065\Assignment2figs" # define some folder 
            #name = "actuator {}".format(i)  # create different image names
            #plt.savefig("{}/{}.png".format(directionary,name))
