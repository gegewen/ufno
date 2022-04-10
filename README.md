# U-FNO - an enhanced Fourier neural operator-based deep-learning model for multiphase flow
In this work, we introduce a model architecture, [U-FNO] (https://www.sciencedirect.com/science/article/pii/S0309170822000562), for solving a dynamic CO<sub>2</sub>-water multiphase flow problem in the context of carbon capture and storage (CCS). The figure below shows that schematic of U-FNO, where we enhances the experssiveness of [Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895) by appending a mini U-Net path to the Fourier layer. 
![model](https://user-images.githubusercontent.com/34537648/160530063-255b53c6-f4db-4ceb-82ba-d6f7c2297ef3.jpg)

## Data sets
The data set is available at: https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing

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
The pre-trained models is available at: https://drive.google.com/drive/folders/1eHTGITZUM55NokoWqaPSzLRoJMIoJQoD?usp=sharing

## Requirements
- [PyTorch 1.8.0](https://pytorch.org/)

## Citation
@article{wen2022u,
  title={U-FNO--An enhanced Fourier neural operator-based deep-learning model for multiphase flow},
  author={Wen, Gege and Li, Zongyi and Azizzadenesheli, Kamyar and Anandkumar, Anima and Benson, Sally M},
  journal={Advances in Water Resources},
  pages={104180},
  year={2022},
  publisher={Elsevier}
}
