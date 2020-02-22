# Hammersmith Hospital using a Philips 3T system
# Guy’s Hospital using a Philips 1.5T system
# Institute of Psychiatry using a GE 1.5T system
import os

import nibabel as nib
import numpy as np

from Utils.MotionUtils.ImageTransform import ImageTransformer
from Utils.kspace.CartesianSampler import CartesianSampler

baseDir = "E:/Workspaces/PhillipsProject/Data/"
t1Path = baseDir + "T1/"
t1Images = os.listdir(t1Path)
t2Path = baseDir + "T2/"
t2Images = os.listdir(t2Path)


def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix. """
    euler = np.asarray(euler, dtype=np.float64).transpose()
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def linearNormalization(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))


def generateMotion(img, voxelRes, maxDisplacementInMillimeter, maxRotInDegree):
    voxelRes = np.asarray(voxelRes)
    maxDisplacementInMillimeter = np.asarray(maxDisplacementInMillimeter)
    maxRotInDegree = np.asarray(maxRotInDegree)

    nT = img.shape[1]

    maxDisplacementInPixel = np.floor(maxDisplacementInMillimeter / voxelRes)
    displacementPixelTrajectory = np.zeros((3, nT))
    rotationDegreeTrajectory = np.zeros((3, nT))
    for i in range(3):
        randomMovement = maxDisplacementInPixel[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, 32]))
        displacementPixelTrajectory[i, :] = np.round(randomMovement)
        randomRotation = maxRotInDegree[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, 32]))
        rotationDegreeTrajectory[i, :] = np.round(randomRotation)

    kspaceSampler = CartesianSampler(img.shape, axes=(0, 2))
    imageTransform = ImageTransformer(img)
    for time in range(nT):
        rotatedImage = imageTransform.rotate_along_axis(rotationDegreeTrajectory[0, time], rotationDegreeTrajectory[1, time], rotationDegreeTrajectory[2, time]
                                                        , displacementPixelTrajectory[0, time], displacementPixelTrajectory[1, time], displacementPixelTrajectory[2, time])
        kspaceSampler.sample(rotatedImage, time)
        imageTransform = ImageTransformer(rotatedImage)
    distortedImageWithMotion = kspaceSampler.getImageAfterSampling()

    pass


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


for imageName in t1Images:
    imgStructure = nib.load(t1Path + imageName)
    voxelSize = imgStructure.header["pixdim"]
    data = imgStructure.get_fdata()
    generateMotion(data, voxelSize[1:4], [1.2, 1.5, 1.4], [10, 20, 30])
    pass
