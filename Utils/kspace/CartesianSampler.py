import numpy as np


class CartesianSampler:
    def __init__(self, shape, axes) -> None:
        super().__init__()
        self.sampledImage = np.zeros(shape, dtype=complex)
        self.axis = axes
        self.distortedImage = None

    def sample(self, image, time):
        fft3Result = np.fft.fftn(image)
        if 0 not in self.axis:
            self.sampledImage[time, :, :] = fft3Result[time, :, :]
        elif 1 not in self.axis:
            self.sampledImage[:, time, :] = fft3Result[:, time, :]
        elif 2 not in self.axis:
            self.sampledImage[:, :, time] = fft3Result[:, :, time]
        print("Cartesian k-space sampling is done in time : {0}".format(time))

    def getSlice(self, time):
        if self.distortedImage is None:
            print("image should be sampled by k-space sampler first")
        else:
            if 0 not in self.axis:
                return self.distortedImage[time, :, :]
            elif 1 not in self.axis:
                return self.distortedImage[:, time, :]
            elif 2 not in self.axis:
                return self.distortedImage[:, :, time]

    def calculateImageAfterSampling(self):
        self.distortedImage = np.round(np.real(np.fft.ifftn(self.sampledImage)))
