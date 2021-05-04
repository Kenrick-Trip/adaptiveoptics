import os
import numpy as np
import time

curpath = os.getcwd()
os.chdir(os.path.dirname(__file__))

import clr
clr.AddReference("System.Runtime.InteropServices")
clr.AddReference("Thorlabs.TLDFM_64.Interop")

import Thorlabs.TLDFM_64.Interop
from System.Text import StringBuilder
from System import Array,Double,Boolean,UInt32
import System.Runtime.InteropServices

os.chdir(curpath)



class ThorlabsDM:
    
    def __init__(self):
        
        self.minValue = 0
        self.maxValue = 200
        
        _,count=Thorlabs.TLDFM_64.Interop.TLDFM.get_device_count(0)
        
        if count == 0:
            raise RuntimeError('No deformable mirror found')
            
        resourceNames=[]
        for i in range(count):
            manufacturer=StringBuilder()
            instrName=StringBuilder()
            serialNumber=StringBuilder()
            resourceName=StringBuilder()

            devAvail = Thorlabs.TLDFM_64.Interop.TLDFM.get_device_information(0, manufacturer,
                                    instrName, serialNumber, False, resourceName)[1]
            
            print(f"Found device: '{instrName.ToString()}' serial: {serialNumber.ToString()}")
            resourceNames.append(resourceName.ToString())

#        resourcelist=rm.FindRsrc(Thorlabs.TLDFM_64.Interop.TLDFM.FindPattern)
        resourceName = resourceNames[0]
        self.dm=Thorlabs.TLDFM_64.Interop.TLDFM(resourceName,True,True)

        self.dm.reset()
        self.dm.enable_hysteresis_compensation(Thorlabs.TLDFM_64.Interop.DevicePart.Both,True)
        
        self.numseg = self.dm.get_segment_count(0)[1]
        self.numtilt = self.dm.get_tilt_count(0)[1]
        
        print(f"DM has {self.numseg} segments and {self.numtilt} tilt channels")

        offsetact=Array[Double]([0.0]*self.numseg)
        offsettiptilt=Array[Double]([0.0]*self.numtilt)

        self.dm.set_voltages(offsetact,offsettiptilt)

    def __len__(self):
        return self.numseg+self.numtilt
    
    def setActuators(self,act):
        act = np.ascontiguousarray(np.clip(act,-1,1) + 1) / 2 * (self.maxValue-self.minValue) + self.minValue
        #print(act)
        assert len(act) == len(self)
        
        voltactarray=Array[Double](act[:self.numseg])
        volttiptiltarray=Array[Double](act[self.numseg:])
        self.dm.set_voltages(voltactarray,volttiptiltarray)

    def getActuators(self):
        voltact=Array[Double]([0.0]*self.numseg)
        volttiptilt=Array[Double]([0.0]*self.numtilt)

        self.dm.get_voltages(voltact,volttiptilt) # function to read voltages.

        voltages=np.zeros(len(self))

        for i in range(self.numseg):
            voltages[i]=voltact[i]

        for i in range(self.numtilt):
            voltages[self.numseg+i]=volttiptilt[i]

        # Rescale to -1 .. 1            
        voltages = (voltages - self.minValue) / (self.maxValue-self.minValue) * 2 - 1
        return voltages

        
    def __enter__(self):
        return self
    
    def close(self):
        if self.dm is not None:
            self.dm.Dispose()
            self.dm = None
        

    def __exit__(self, *args):
        self.close()
        
    
if __name__=="__main__":
    with ThorlabsDM() as dm:
        num=dm.numseg
        al=len(dm)
        act=np.zeros([al])
        act[:num]=0.0
        act=np.ones([al])*0.9
        l=np.linspace(-0.9,0.9,5)
       
        while True:
            for i in range(5):
                act[:num]=l[i]*-1
                dm.setActuators(act)
                
                time.sleep(0.2)
                
            for i in range(5):
                act[:num]=l[i]
                dm.setActuators(act)
                
                time.sleep(0.2)
