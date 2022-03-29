# U-FNO - an enhanced Fourier neural operator-based deep-learning model for multiphase flow
## Model architecture
Code in `ufno.py`
![model](https://user-images.githubusercontent.com/34537648/160530063-255b53c6-f4db-4ceb-82ba-d6f7c2297ef3.jpg)

## Data sets
The data set is available at https://drive.google.com/drive/folders/1T0Y_IVfeOszv0MDfR-bU0E7PoMKcZSPt?usp=sharing

#### Train set (n = 4,500):
- input: `sg_train_a.pt`, output: `sg_train_u.pt`
- input: `dP_train_a.pt`, output: `dP_train_u.pt`

#### Validation set (n = 500):
- input: `sg_val_a.pt`, output: `sg_val_u.pt`
- input: `dP_train_a.pt`, output: `dP_train_u.pt`

#### Test set (n = 500):
- input: `sg_test_a.pt`, output: `sg_test_u.pt`
- input: `dP_test_a.pt`, output: `dP_test_u.pt`

## Pre-trained models
The pre-trained models is available at 

## Training example

## Evaluation example


## Requirements
- [PyTorch 1.8.0](https://pytorch.org/)
