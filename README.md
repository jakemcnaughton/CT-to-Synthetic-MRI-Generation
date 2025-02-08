# Synthetic Brain MRI Generation from CT Scans Using Deep Learning

This repository collates the models developed in [our paper](https://www.mdpi.com/2673-7426/3/3/50) in which we use CNNs to translate CTs to MRIs for stroke patients. MRIs have specific benefits over CTs: they have better contrast and detail of soft tissue and lesions and registration of brain scans is often registered to an MRI atlas, making an MRI the preferable model to have available. See our recent [review](https://www.mdpi.com/2306-5354/10/9/1078) to compare different types of medical image generation.

## How to Use this Repo
There are *8* different architectures in the Models folder. To run a model, first select if you want to run it 3D, 3D patch-based, or 2D. Then select the respective file for running: main3D.py, main_patch.py, main2D.py

Import ONLY the model you want to use.


## Preprocessing
We aligned the CT and MRI of each patient together and then aligned everything to the MNI152 atlas before performing brain extraction.
![alt text](https://github.com/jakemcnaughton/CT-to-Synthetic-MRI-Generation/blob/main/Pipeline.png)

## Training
Set Test = False

Select GPUs (export CUDA_VISIBLE_DEVICES= x, y, z, ...)

Set NAME to the name of your model

Run "python main3D.py"


## Testing
Set Test = True

Run "python main3D.py"


## Evaluation
"eval.py" calculate SSIM, MAE, MSE, and PSNR over the non-zero pixels of the results.

"toimage.py" lets you save Image slices of your nifti files.

## Citing
If you use this code please cite the following papers.
```bibtex
 @Article{bioengineering10091078,
AUTHOR = {McNaughton, Jake and Fernandez, Justin and Holdsworth, Samantha and Chong, Benjamin and Shim, Vickie and Wang, Alan},
TITLE = {Machine Learning for Medical Image Translation: A Systematic Review},
JOURNAL = {Bioengineering},
VOLUME = {10},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {1078},
URL = {https://www.mdpi.com/2306-5354/10/9/1078},
ISSN = {2306-5354},
DOI = {10.3390/bioengineering10091078}
 }
```

```bibtex
@Article{biomedinformatics3030050,
AUTHOR = {McNaughton, Jake and Holdsworth, Samantha and Chong, Benjamin and Fernandez, Justin and Shim, Vickie and Wang, Alan},
TITLE = {Synthetic MRI Generation from CT Scans for Stroke Patients},
JOURNAL = {BioMedInformatics},
VOLUME = {3},
YEAR = {2023},
NUMBER = {3},
PAGES = {791--816},
URL = {https://www.mdpi.com/2673-7426/3/3/50},
ISSN = {2673-7426},
DOI = {10.3390/biomedinformatics3030050}
}
```
