<!-- The code is written with inspiration from Deep Learning in Production Book, adapted for PyTorch -->
# SDF TO NURBS vice-versa

This code is used for training and  testing the mapping between signed distance and NURBS, vice-versa


## Requirements

### Install required packages

The training and inference both are performed with CUDA enabled devices

Run

```
conda env create -f environment.yml
conda activate spline
```


## Usage

1. Create a ```lhs_ctrlpts.npy``` which contains the variation of the control points
2. Similarly you need images2 file which contains corresponding signed distance field


3. Run training script with ```python main.py $(pwd)/config.txt```. with "mode" set to train



4. Run inference script with ```python main.py $(pwd)/config.txt``  with "mode" set to test

## Validation 
```validation```  folder contains dummy files to test the code for reproducibility


