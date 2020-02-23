# MRI_Motion_Classification
MRI_Motion_Classification

Generating synthetic motion on MRI images based on Cartesian k-space sampler: Utils/MotionUtils/GenerateMotion.py
More k-space sampling trajectory can be designed just by interfacing CartesianSampler.py

The generated motion is slice-dependent. In other words, for each slice it rotate the rigid 3d volume and apply k-space sampling in appropriate time.



