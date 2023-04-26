# Synthetic Brain MRI Generation from CT Scans Using Deep Learning

This repository collates the models I developed for my masters thesis. The purpose of this development was to develop models that could accurately produce an MRI from a CT scan for suspected stroke patients. MRIs have specific benefits over CTs: they have better contrast and detail of soft tissue and lesions and registration of brain scans is often registered to an MRI atlas, making an MRI the preferable model to have available.

## How to Use this Repo
There are *8* different architectures in the Models folder. To run a model, first select if you want to run it 3D, 3D patch-based, or 2D. Then select the respective file for running: main3D.py, main_patch.py, main2D.py

Import ONLY the model you want to use.


## Pre-process your data:
My pipeline was to 


## To Train:
Set Test = False
Select GPUs (export CUDA_VISIBLE_DEVICES= x, y, z, ...)
Set ... to the name of your model
Run "python main3D.py"
