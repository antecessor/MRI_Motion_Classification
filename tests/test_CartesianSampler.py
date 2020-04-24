import os
from unittest import TestCase

import nibabel as nib
import pandas as pd


class TestGetImageResolutions(TestCase):
    def test_getImageResolutionsForT1(self):
        baseDir = "E:/Workspaces/PhillipsProject/Data/"
        t1Path = baseDir + "T1/"
        t1Images = os.listdir(t1Path)
        t2Path = baseDir + "T2/"
        t2Images = os.listdir(t2Path)
        tesla15GEVoxels = []
        tesla3Voxels = []
        tesla15PHVoxels = []
        for imageName in t1Images:
            imgStructure = nib.load(t1Path + imageName)
            voxelSize = imgStructure.header["pixdim"][1:4]
            if imageName.__contains__("HH"):
                tesla3Voxels.append(voxelSize)
            elif imageName.__contains__("Guys"):
                tesla15PHVoxels.append(voxelSize)
            elif imageName.__contains__("IOP"):
                tesla15GEVoxels.append(voxelSize)

            print("image : {} is processing".format(imageName))

        GE15Tesla = pd.DataFrame(tesla15GEVoxels)
        PH15Tesla = pd.DataFrame(tesla15PHVoxels)
        PH3Tesla = pd.DataFrame(tesla3Voxels)

        GE15Tesla.to_excel("T1_1.5Tesla_GE.xlsx")
        PH15Tesla.to_excel("T1_1.5Tesla_PH.xlsx")
        PH3Tesla.to_excel("T1_3Tesla_PH.xlsx")

        pass

    def test_getImageResolutionsForT2(self):
        baseDir = "E:/Workspaces/PhillipsProject/Data/"
        t1Path = baseDir + "T1/"
        t1Images = os.listdir(t1Path)
        t2Path = baseDir + "T2/"
        t2Images = os.listdir(t2Path)
        tesla15GEVoxels = []
        tesla3Voxels = []
        tesla15PHVoxels = []
        for imageName in t2Images:
            imgStructure = nib.load(t2Path + imageName)
            voxelSize = imgStructure.header["pixdim"][1:4]
            if imageName.__contains__("HH"):
                tesla3Voxels.append(voxelSize)
            elif imageName.__contains__("Guys"):
                tesla15PHVoxels.append(voxelSize)
            elif imageName.__contains__("IOP"):
                tesla15GEVoxels.append(voxelSize)

            print("image : {} is processing".format(imageName))

        GE15Tesla = pd.DataFrame(tesla15GEVoxels)
        PH15Tesla = pd.DataFrame(tesla15PHVoxels)
        PH3Tesla = pd.DataFrame(tesla3Voxels)

        GE15Tesla.to_excel("T2_1.5Tesla_GE.xlsx")
        PH15Tesla.to_excel("T2_1.5Tesla_PH.xlsx")
        PH3Tesla.to_excel("T2_3Tesla_PH.xlsx")

        pass
