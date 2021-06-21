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
    #[wo, ho] = [416, 850]
    #print(wo)
    #print(ho)
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
        
        n =len(dm)
        
        for i in range(n):
            A = np.zeros(len(dm))
            # poke actuator
            A[i] = 1
            dm.setActuators(A)
            
            img = grabframes(5,1)
            w = 200
            h = 200
            
            fig = zoomImage(img[-1], w, h)
            
            plt.figure()
            plt.imshow(fig)
