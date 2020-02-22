import numpy as np


class CartesianSampler:
    def __init__(self, shape, axes) -> None:
        super().__init__()
        self.sampledImage = np.zeros(shape, dtype=complex)
        self.axis = axes

    def sample(self, image, time):
        fft3Result = np.fft.fftn(image, axes=self.axis)
        if 0 not in self.axis:
            self.sampledImage[time, :, :] = fft3Result[time, :, :]
        elif 1 not in self.axis:
            self.sampledImage[:, time, :] = fft3Result[:, time, :]
        elif 2 not in self.axis:
            self.sampledImage[:, :, time] = fft3Result[:, :, time]
        print("Cartesian k-space sampling is done in time : {0}".format(time))

    def getImageAfterSampling(self):
        return np.real(np.fft.ifftn(self.sampledImage, axes=self.axis))
