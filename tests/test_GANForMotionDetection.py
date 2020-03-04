import os
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DeepLearning.CycleGAN import CycleGAN
from Utils.DataUtils.LoadingUtils import readImage


class TestGANForMotionDetection(TestCase):

    def createPairImages(self, dataFrame):
        X = []
        Y = []
        for index, row in dataFrame.iterrows():
            if row[0] != 0:
                name = row[1]
                imageWithMotion = row[3]
                number = row[2]
                df1 = dataFrame[(dataFrame.motionNorm == 0) & (dataFrame.name == name) & (dataFrame.number == number)]
                imageWithoutMotion = df1["image"].values[0]
                if imageWithoutMotion.shape != (256, 256) or imageWithMotion.shape != (256, 256):
                    continue
                X.append(imageWithMotion)
                Y.append(imageWithoutMotion)
        return X, Y

    def test_trainingGAN(self):
        baseDir = "E:\Workspaces\PhillipsProject\Data\generated/"
        images = os.listdir(baseDir)
        dataFrame = pd.DataFrame(columns=["motionNorm", "name", "number", "image"])
        motionValue = []
        names = []
        numbers = []
        imageMats = []
        for image in images:
            imageMat = readImage(baseDir + image, show=False)

            params = image.split("_")
            displacementNorm = float(params[0])
            rotationNorm = float(params[1])
            motionValue.append(np.sqrt(np.power(displacementNorm, 2) + np.power(rotationNorm, 2)))
            names.append(params[2])
            numbers.append(params[3])
            imageMats.append(imageMat)
            print("load image {}".format(image))
        dataFrame['motionNorm'] = motionValue
        dataFrame['name'] = names
        dataFrame['number'] = numbers
        dataFrame['image'] = imageMats

        X, Y = self.createPairImages(dataFrame)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
        X_train = np.reshape(X_train, [-1, X_train[0].shape[0], X_train[0].shape[0], 1])
        X_test = np.reshape(X_test, [-1, X_test[0].shape[0], X_test[0].shape[0], 1])
        y_train = np.reshape(y_train, [-1, y_train[0].shape[0], y_train[0].shape[0], 1])
        y_test = np.reshape(y_test, [-1, y_test[0].shape[0], y_test[0].shape[0], 1])

        gan = CycleGAN()
        gan.train(x_train=X_train, y_train=y_train, epochs=200, batch_size=1, sample_interval=200)
