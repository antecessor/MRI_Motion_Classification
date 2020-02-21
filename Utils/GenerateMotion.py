# Hammersmith Hospital using a Philips 3T system
# Guy’s Hospital using a Philips 1.5T system
# Institute of Psychiatry using a GE 1.5T system
import os

import nibabel as nib
import numpy as np
from noise import pnoise1

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


def generateMotion(img, maxDisp, maxRot):
    fft3Result = np.fft.fftn(img, axes=(1, 2))
    nT = fft3Result.shape[1]

    swallowFrequency = 2  # number of swallowing events in scan
    swallowMagnitude = [2, 2]  # first is translations, second is rotations
    suddenFrequency = 2  # number of sudden movements
    suddenMagnitude = [2, 2]  # first is translations, second is rotations

    fitpars = np.zeros((6, nT))
    fitpars[0, :] = maxDisp * pnoise1(nT)
    fitpars[1, :] = maxDisp * pnoise1(nT)
    fitpars[2, :] = maxDisp * pnoise1(nT)
    fitpars[3, :] = maxRot * pnoise1(nT)
    fitpars[4, :] = maxRot * pnoise1(nT)
    fitpars[5, :] = maxRot * pnoise1(nT)

    # add swallowing movement in z-axis
    swallowTraceBase = np.exp(-np.linspace(0, 1e2, nT))
    swallowTrace = np.zeros((1, nT))
    for iS in range(swallowFrequency):
        swallowTrace = swallowTrace + np.roll(swallowTraceBase, np.random.randint(0, nT, 1))
    fitpars[2, :] = fitpars[2, :] + swallowMagnitude[0] * swallowTrace
    fitpars[3, :] = fitpars[3, :] + swallowMagnitude[1] * swallowTrace

    fitMats = euler2mat(fitpars[3:6, :])
    # fitMats[0:3, 3, :] = fitpars[0: 3, :]

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
    generateMotion(data, 1.2, 1.4)
    pass
