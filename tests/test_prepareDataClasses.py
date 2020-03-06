import os
from unittest import TestCase

import numpy as np
import pandas as pd
from PIL import Image
from keract import keract, get_activations
from keras.engine.saving import load_model

from Utils.DataUtils.LoadingUtils import readImage


class TestPrepareDataForTraining(TestCase):
    def test_getActivationLayers(self):
        model = load_model('motionClassificationModel.h5')
        x = np.random.uniform(size=(256, 256))
        activations = get_activations(model, x, auto_compile=True)
        image1 = "E:\Workspaces\PhillipsProject\Data\generated\\2.45_2.45_IXI002-Guys-0828-T1_56.tiff"
        image2 = "E:\Workspaces\PhillipsProject\Data\generated\\2.45_3.0_IXI012-HH-1211-T2_54"
        mageMat1 = readImage(image1, show=False)
        mageMat2 = readImage(image2, show=False)

        keract.display_heatmaps(activations, mageMat1, save=False)

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
            # imageMat = imageMat - imageMat[0, 0]
            # cv2.normalize(imageMat, imageMat, 0, 255, cv2.NORM_MINMAX)
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

        for row in dataFrame.iterrows():
            motionNorm = np.sqrt(np.power(np.asarray(row[1].displacementNorm), 2) + np.power(np.asarray(row[1].rotationNorm), 2))
            im = Image.fromarray(row[1].image)

            if motionNorm == 0:
                saveDir = baseDir + "M0/"
            elif motionNorm <= 2.5 and motionNorm > 0:
                saveDir = baseDir + "M1/"
            elif motionNorm <= 3.5 and motionNorm > 2.5:
                saveDir = baseDir + "M2/"
            elif motionNorm <= 4.5 and motionNorm > 3.5:
                saveDir = baseDir + "M3/"
            elif motionNorm > 4.5:
                saveDir = baseDir + "M4/"

            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            imageName = saveDir + "{0}.tiff".format(row[0])
            im.save(imageName)
