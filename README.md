# Debiasing surgeon: fantastic weights and how to find them

<!-- [![paper](https://img.shields.io/badge/ICCV-paper-blue)](link) -->
[![arXiv](https://img.shields.io/badge/arXiv-2305.03691-b31b1b.svg)](https://arxiv.org/abs/2403.14200)

The official repository. Please cite as
```
@InProceedings{Nahon_2024_ECCV,
    author    = {Nahon, R\'emi and De Moura Matos, Ivan Luiz and Nguyen, Van-Tam and Tartaglione, Enzo},
    title     = {Debiasing surgeon: fantastic weights and how to find them},
    booktitle = {Proceedings of the IEEE/CVF European Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2024},
  
}
```
This is an example implementation of  *Debiasing surgeon: fantastic weights and how to find them.*


## Installations

If you want to work with conda :

```bash
conda create -n ffw python=3.9
conda activate ffw
pip3 install -r pip_requirements.txt
```

If not, simply install the requirement with :

```bash
pip3 install -r pip_requirements.txt
```

## Run our method on Biased-MNIST

1. In a terminal, get inside the ffw folder

2. Run : python3 main.py

### Possible extra arguments #####

- --dev : to specify the device you want to work on (ex: cpu or cuda:0)
- --dataset : to specify the dataset you want to use FFW on
- other extra arguments detailed in utils/configs.py