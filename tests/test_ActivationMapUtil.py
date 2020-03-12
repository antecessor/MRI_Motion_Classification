from unittest import TestCase

import numpy as np
from keract import get_activations, keract
from tensorflow.python.keras.models import load_model

from Utils.ActivationMapUtils.ActivationMapUtil import visualize_class_activation_map
from Utils.DataUtils.LoadingUtils import readImage


class Test(TestCase):
    def test_getActivationLayers(self):
        model = load_model('../motionClassificationModel.h5')
        x = np.random.uniform(size=(256, 256))
        x = np.reshape(x, (-1, 256, 256, 1))
        activations = get_activations(model, x, auto_compile=True)
        image1 = "E:\Workspaces\PhillipsProject\Data\generated\\2.45_2.45_IXI002-Guys-0828-T1_56.tiff"
        image2 = "E:\Workspaces\PhillipsProject\Data\generated\\2.45_3.0_IXI012-HH-1211-T2_54.tiff"
        mageMat1 = readImage(image1, show=False)
        mageMat2 = readImage(image2, show=False)
        mageMat1 = np.reshape(mageMat1, (-1, 256, 256, 1))
        keract.display_heatmaps(activations, mageMat1, save=True)

    def test_visualize_class_activation_map(self):
        visualize_class_activation_map("E:\Workspaces\PhillipsProject\Python\motionClassificationModel.h5", "E:\Workspaces\PhillipsProject\Data\generated\\0.0_0.0_IXI002-Guys-0828-T1_50.tiff", "./")
