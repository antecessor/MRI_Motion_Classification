import os
from unittest import TestCase

import cv2
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
        return X, Y, displacementNorm, rotationNorm

    def test_trainingGAN(self):
        baseDir = "E:\Workspaces\PhillipsProject\Data\generated/"
        images = os.listdir(baseDir)
        dataFrame = pd.DataFrame(columns=["displacementNorm", "rotationNorm", "name", "number", "image"])
        names = []
        numbers = []
        imageMats = []
        displacementNormValues = []
        rotationNormValue = []
        for image in images:
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
        dataFrame['displacementNorm'] = displacementNormValues
        dataFrame['rotationNorm'] = rotationNormValue
        dataFrame['name'] = names
        dataFrame['number'] = numbers
        dataFrame['image'] = imageMats

        X, Y, displacementNorm, rotationNorm = self.createPairImages(dataFrame)
        information = pd.DataFrame(columns=["RMS of rotation", "RMS of displacement", "SSIM", "PSNR"])
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
        DA = DataAnalysisUtils()
        # DA.plotCorrelationHeatMap(information)
        DA.plotScatterAllFeatures(information)
        pass
