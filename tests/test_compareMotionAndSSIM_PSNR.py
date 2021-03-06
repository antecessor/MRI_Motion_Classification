import os
from unittest import TestCase

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio

from Utils.DataAnalysis import DataAnalysisUtils
from Utils.DataUtils.LoadingUtils import readImage


class TestCompareMotionWithSSIMPSNR(TestCase):

    def createPairImages(self, dataFrame):

        X = []
        Y = []
        rotationNorm = []
        displacementNorm = []
        t1t2 = []
        for index, row in dataFrame.iterrows():
            if row[0] != 0 and row[1] != 0:
                name = row[2]
                imageWithMotion = row[4]
                number = row[3]
                df1 = dataFrame[(dataFrame.displacementNorm == 0) & (dataFrame.name == name) & (dataFrame.number == number)]
                imageWithoutMotion = df1["image"].values[0]
                if imageWithoutMotion.shape != (256, 256) or imageWithMotion.shape != (256, 256):
                    continue
                X.append(imageWithMotion)
                Y.append(imageWithoutMotion)
                rotationNorm.append(row[1])
                displacementNorm.append(row[0])
                t1t2.append(row[-1])
        return X, Y, displacementNorm, rotationNorm, t1t2

    def test_jointHistogram(self):
        baseDir = "E:\Workspaces\PhillipsProject\Data\generated/"
        images = os.listdir(baseDir)

        imageT1 = []
        imageT2 = []
        count = 1
        for image in images:
            if image.__contains__("T1"):
                try:
                    if count == 100:
                        break
                    count = count + 1
                    imageMat = readImage(baseDir + image, show=False)
                    imageMat = imageMat - imageMat[0, 0]
                    cv2.normalize(imageMat, imageMat, 0, 255, cv2.NORM_MINMAX)
                    cv2.resize(imageMat, (256, 256), imageMat)
                    # imageMat = np.diff(imageMat)
                    imageT1.extend(np.asarray(imageMat[:]).ravel())
                except:
                    pass
        count = 1
        for image in images:
            if image.__contains__("T2"):
                try:
                    if count == 100:
                        break
                    count = count + 1
                    imageMat = readImage(baseDir + image, show=False)
                    # imageMat = imageMat - imageMat[0, 0]
                    cv2.normalize(imageMat, imageMat, 0, 255, cv2.NORM_MINMAX)
                    cv2.resize(imageMat, (256, 256), imageMat)
                    # imageMat = np.diff(imageMat)
                    imageT2.extend(np.asarray(imageMat[:]).ravel())
                except:
                    pass
        fig2 = plt.figure()
        imageT1 = np.asarray(imageT1)
        imageT2 = np.asarray(imageT2)
        plt.hist2d(imageT1[range(len(imageT2))], imageT2, bins=100, norm=mcolors.PowerNorm(.8), cmax=1000)
        plt.xlabel('T1')
        plt.ylabel('T2')
        cbar = plt.colorbar()
        plt.tight_layout()
        cbar.ax.set_ylabel('Counts')
        plt.savefig("jointHistT1T2.png")
        pass

    def test_ImageQuality(self):
        baseDir = "E:\Workspaces\PhillipsProject\Data\generated/"
        images = os.listdir(baseDir)
        dataFrame = pd.DataFrame(columns=["displacementNorm", "rotationNorm", "name", "number", "image"])
        names = []
        numbers = []
        imageMats = []
        displacementNormValues = []
        rotationNormValue = []
        t1t2 = []
        for image in images:
            if image.__contains__("T1"):
                try:
                    imageMat = readImage(baseDir + image, show=False)
                    imageMat = imageMat - imageMat[0, 0]
                    cv2.normalize(imageMat, imageMat, 0, 255, cv2.NORM_MINMAX)
                    params = image.split("_")
                    displacementNorm = float(params[0])
                    rotationNorm = float(params[1])
                    displacementNormValues.append(displacementNorm)
                    rotationNormValue.append(rotationNorm)
                    names.append(params[2])
                    numbers.append(params[3])
                    imageMats.append(imageMat)
                    print("load image {}".format(image))
                    t1t2.append(1)
                except:
                    pass

        for image in images:
            if image.__contains__("T2"):
                try:
                    imageMat = readImage(baseDir + image, show=False)
                    imageMat = imageMat - imageMat[0, 0]
                    cv2.normalize(imageMat, imageMat, 0, 255, cv2.NORM_MINMAX)
                    params = image.split("_")
                    displacementNorm = float(params[0])
                    rotationNorm = float(params[1])
                    displacementNormValues.append(displacementNorm)
                    rotationNormValue.append(rotationNorm)
                    names.append(params[2])
                    numbers.append(params[3])
                    imageMats.append(imageMat)
                    t1t2.append(2)
                    print("load image {}".format(image))
                except:
                    pass
        dataFrame['displacementNorm'] = displacementNormValues
        dataFrame['rotationNorm'] = rotationNormValue
        dataFrame['name'] = names
        dataFrame['number'] = numbers
        dataFrame['image'] = imageMats
        dataFrame['T1T2'] = t1t2

        X, Y, displacementNorm, rotationNorm, t1t2 = self.createPairImages(dataFrame)
        information = pd.DataFrame(columns=["RMS of rotation", "RMS of displacement", "SSIM", "PSNR", "T1T2"])
        ssimValues = []
        psnrValues = []
        for index, image in enumerate(X):
            ssimValues.append(ssim(image, Y[index], data_range=image.max() - image.min()))
            psnrValues.append(peak_signal_noise_ratio(Y[index], image, data_range=image.max() - image.min()))

        information["RMS of rotation"] = np.asarray(rotationNorm, dtype=float)
        information["RMS of displacement"] = np.asarray(displacementNorm, dtype=float)
        information["SSIM"] = np.asarray(ssimValues, dtype=float)
        information["PSNR"] = np.asarray(psnrValues, dtype=float)
        information["|Motion|"] = np.sqrt(np.power(np.asarray(rotationNorm), 2) + np.power(np.asarray(displacementNorm), 2))
        information["T1T2"] = np.asarray(t1t2, dtype=float)
        DA = DataAnalysisUtils()
        # DA.plotCorrelationHeatMap(information)
        DA.plotPairAllFeatureByHue(information, "T1T2")
        pass
