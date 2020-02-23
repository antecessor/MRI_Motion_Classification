# Hammersmith Hospital using a Philips 3T system
# Guy’s Hospital using a Philips 1.5T system
# Institute of Psychiatry using a GE 1.5T system
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

from Utils.MotionUtils.ImageTransform import ImageTransformer
from Utils.kspace.CartesianSampler import CartesianSampler

baseDir = "E:/Workspaces/PhillipsProject/Data/"
genDir = "E:/Workspaces/PhillipsProject/Data/generated/"
t1Path = baseDir + "T1/"
t1Images = os.listdir(t1Path)
t2Path = baseDir + "T2/"
t2Images = os.listdir(t2Path)

fig, axes = plt.subplots(1, 2)


def showSlice(slices):
    for index, slice in enumerate(slices):
        axes[index].imshow(slice, cmap="gray", origin="lower")


def saveSlice(slice, rotationDegreeTrajectory, displacementPixelTrajectory, suffix=''):
    rotationVal = np.linalg.norm(rotationDegreeTrajectory)
    displacementVal = np.linalg.norm(displacementPixelTrajectory)
    normalizedImg = np.asarray(np.round(linearNormalization(slice) * 255), dtype=np.uint8)
    im = Image.fromarray(normalizedImg)
    imageName = "{0}{1}_{2}_{3}.tiff".format(genDir, round(displacementVal, 2), round(rotationVal, 2), suffix)
    im.save(imageName)


def linearNormalization(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))


def findANumberWithMod0(primaryNum):
    firstNum = int(primaryNum / 2)
    while firstNum > 2:
        if primaryNum % firstNum == 0:
            return firstNum


def generateMotion(img, voxelRes, maxDisplacementInMillimeter, maxRotInDegree, primaryAxis=2, imageNameSuffix=''):
    voxelRes = np.asarray(voxelRes)
    maxDisplacementInMillimeter = np.asarray(maxDisplacementInMillimeter)
    maxRotInDegree = np.asarray(maxRotInDegree)

    nT = img.shape[primaryAxis]
    axes = (1, 2)
    if primaryAxis == 0:
        axes = (1, 2)
    elif primaryAxis == 1:
        axes = (0, 2)
    elif primaryAxis == 2:
        axes = (0, 1)

    maxDisplacementInPixel = np.floor(maxDisplacementInMillimeter / voxelRes)
    displacementPixelTrajectory = np.zeros((3, nT))
    rotationDegreeTrajectory = np.zeros((3, nT))
    for i in range(3):
        randomMovement = maxDisplacementInPixel[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, findANumberWithMod0(nT)]))
        displacementPixelTrajectory[i, :] = np.round(randomMovement)
        randomRotation = maxRotInDegree[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, findANumberWithMod0(nT)]))
        rotationDegreeTrajectory[i, :] = np.round(randomRotation)

    kspaceSampler = CartesianSampler(img.shape, axes=axes)
    kspaceSamplerWithoutMovement = CartesianSampler(img.shape, axes=axes)
    kspaceSamplerWithoutMovement.distortedImage = img

    imageTransform = ImageTransformer(img)
    for time in range(nT):
        rotatedImage = imageTransform.rotate_along_axis(rotationDegreeTrajectory[0, time], rotationDegreeTrajectory[1, time], rotationDegreeTrajectory[2, time]
                                                        , displacementPixelTrajectory[0, time], displacementPixelTrajectory[1, time], displacementPixelTrajectory[2, time])
        kspaceSampler.sample(rotatedImage, time)
        imageTransform = ImageTransformer(img)
    kspaceSampler.calculateImageAfterSampling()
    for time in range(int(nT / 3), nT - int(nT / 3), 1):
        slices = [kspaceSampler.getSlice(time), kspaceSamplerWithoutMovement.getSlice(time)]
        # showSlice(slices)
        saveSlice(slices[0], rotationDegreeTrajectory[:, time], displacementPixelTrajectory[:, time], imageNameSuffix.replace(".nii.gz", "_" + str(time)))
        saveSlice(slices[1], [0, 0, 0], [0, 0, 0], imageNameSuffix.replace(".nii.gz", "_" + str(time)))


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


# for imageName in t1Images:
#     imgStructure = nib.load(t1Path + imageName)
#     voxelSize = imgStructure.header["pixdim"]
#     data = imgStructure.get_fdata()
#     generateMotion(data, voxelSize[1:4], maxDisplacementInMillimeter=[3, 3, 3], maxRotInDegree=[3, 3, 3], primaryAxis=2, imageNameSuffix=imageName)

for imageName in t2Images:
    imgStructure = nib.load(t2Path + imageName)
    voxelSize = imgStructure.header["pixdim"]
    data = imgStructure.get_fdata()
    print("image : {} is processing".format(imageName))
    generateMotion(data, voxelSize[1:4], maxDisplacementInMillimeter=[3, 3, 3], maxRotInDegree=[3, 3, 3], primaryAxis=0, imageNameSuffix=imageName)
