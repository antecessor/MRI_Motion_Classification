# Hammersmith Hospital using a Philips 3T system
# Guy’s Hospital using a Philips 1.5T system
# Institute of Psychiatry using a GE 1.5T system
import os

import nibabel as nib

baseDir = "E:/Workspaces/PhillipsProject/Data/"
t1Path = baseDir + "T1/"
t1Images = os.listdir(t1Path)
t2Path = baseDir + "T2/"
t2Images = os.listdir(t2Path)

for imageName in t1Images:
    img = nib.load(t1Path + imageName)
    voxelSize = img.header["pixdim"]
    data = img.get_fdata()
    pass
