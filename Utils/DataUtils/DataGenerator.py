import os

import cv2
import numpy as np

from Utils.DataUtils.LoadingUtils import readImage

baseDir = "E:/Workspaces/PhillipsProject/Data"
T2W_15T = baseDir + "/T2_1.5T/"
T2W_3T = baseDir + "/T2_3T/"
T1W_15T = baseDir + "/T1_1.5T/"
T1W_3T = baseDir + "/T1_3T/"

selectedDataPath = T2W_15T

motionSeverities = ["M0/", "M1/", "M2/", "M3/", "M4/"]
imageNames = []
classElementsSize = []
for motionSeverity in motionSeverities:
    listdir = os.listdir(selectedDataPath + motionSeverity)
    classElementsSize.append(len(listdir))
    imageNames.append(listdir)

wholeIndex = list(np.cumsum(classElementsSize))
wholeIndex.insert(0, 0)


def getClasses():
    return ["without motion", "small motion", "mild motion", "moderate motion", "severe motion"]


def getImageAndClasses(i, show=False):
    selectedClass = 0
    for indexForEachClass, value in enumerate(wholeIndex):
        if value > i:
            selectedClass = indexForEachClass - 1
            break

    imageName = selectedDataPath + motionSeverities[selectedClass] + imageNames[selectedClass][i - wholeIndex[selectedClass]]
    image = readImage(imageName, show=False)
    image = cv2.resize(image, (256, 256))

    target = np.zeros((1, len(motionSeverities)))
    target[0, selectedClass] = 1
    return image, target[0]


def getLen():
    return wholeIndex[-1]
