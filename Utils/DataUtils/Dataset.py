import numpy as np
from sklearn.model_selection import train_test_split

from Utils.DataUtils.DataGenerator import getImageAndClasses, wholeIndex


class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, train="train", augmentation=None, preprocessing=None, test_size=.3):
        # convert str names to class values on masks
        self.train = train
        labels = np.zeros((1, wholeIndex[-1]), dtype=int)[0]
        for i in range(len(wholeIndex) - 1):
            labels[[range(wholeIndex[i], wholeIndex[i + 1])]] = int(i)
        self.train_filter, self.test_filter, _, _ = train_test_split(list(range(wholeIndex[-1])), labels, test_size=test_size, stratify=labels, shuffle=True)
        _, _, self.validation_filter, self.test_filter = train_test_split(list(range(len(self.test_filter))), self.test_filter, test_size=.33)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        if self.train.__contains__("train"):
            image, label = getImageAndClasses(self.train_filter[i])
        elif self.train.__contains__("test"):
            image, label = getImageAndClasses(self.test_filter[i])
        elif self.train.__contains__("validation"):
            ind = np.random.random_integers(0, len(self.validation_filter) - 1, 1)[0]
            image, label = getImageAndClasses(self.validation_filter[ind])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply augmentations
        if self.augmentation:
            pass

        # apply preprocessing
        if self.preprocessing:
            pass

        return image, label

    def __len__(self):
        if self.train:
            return len(self.train_filter)
        else:
            return len(self.test_filter)
