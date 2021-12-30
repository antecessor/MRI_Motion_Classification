# MRI_Motion_Classification
MRI_Motion_Classification

Generating synthetic motion on MRI images based on Cartesian k-space sampler: Utils/MotionUtils/GenerateMotion.py
More k-space sampling trajectory can be designed just by interfacing CartesianSampler.py

The generated motion is slice-dependent. In other words, for each slice it rotate the rigid 3d volume and apply k-space sampling in appropriate time.

For training the model fix directory in DataGenerator.py

## Reference
Mohebbian, M., Walia, E., Habibullah, M., Stapleton, S. and Wahid, K.A., 2021. Classifying MRI motion severity using a stacked ensemble approach. Magnetic Resonance Imaging, 75, pp.107-115.



