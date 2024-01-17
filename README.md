# TorchSparse for the AT-TPC 

*The following repository is forked from a private repository. It will not always be updated to the latested working version*

TorchSparse is a high-performance neural network library for point cloud processing.

***

# 3D Sparse Convolution with Minkowski Architecture Model in TorchSparse

## Contents

In this directory, there are three folders:

attpc-torchsparse/

├── notebooks/

├── python/

└── shell/


The `notebooks` folder contains a notebook that visualizes the Mg22 data.

The `python` folder contains the python scripts needed for data proccessing and model training and evaluation, as well as plot generation. 

The `shell` folder contains shell scripts that run the data and training pipelines.
> **_NOTE:_**  You will need to manually go into each shell file and change the anaconda enviroment name from *sparse* to whatever you named your enivroment.

## Usage

The data processing and training pipelines are very straightforward to use. Before running any code, however, ensure the data is on your local system. The data the scripts use is Mg22 + α reaction simulated data. For details on how to get the data on your local system, contact Dr. Kuchera or Dr. Ramanujan.

This code is only to be used with simulated data.

`data_processing.sh` takes the .h5 file and converts it into numpy arrays for model training. It is important to understand what the data looks like, so I recommend using the notebook to visualize the structure of the data in order to make changes to the data processing scripts where needed.

`training.sh` is a collection of several Python scripts that carry out the training pipeline, starting from model training, to loss curve plotting, model evaluation, and confusion matrix creation. When the code is run, there will be a new folder, training, created that will store all the training plots and statistics. This pipeline is for point classification, not track classification.

## Installation

The code uses TorchSparse v2.1.0. Alternative installation steps are also provided through the original repo from MIT, but with the following code and version of TorchSparse, the installation is tested to work with no issues as of 12/12/2023. First, you will need to clone the repo and create a conda virtual enviroment:

```
conda create --name myenv python=3.8
```

Once you have your conda env setup, you will have to install the correct PyTorch version:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Once that has been installed, you will then need to run a bash script for the installation of TorchSparse v2.1.0. The script is `installTorchSparse.sh`. Make sure to change the conda enviroment inside the shell script to your conda enviroment name. 

```
sbatch installTorchSparse.sh
```

This installation process is different to the one suggested by the MIT team, but if this process does not work, please reference their [installation guide](https://torchsparse-docs.github.io/getting_started/installation.html) for more details.

## Required packages:
If you follow the install instructions, you should only need to manually install a few packages for the python scripts to run.

- click
- h5py
- jupyterlab
- matplotlib
- scipy
- threadpoolctl
- torch_scatter

```
conda install click h5py jupyterlab matplotlib scipy
conda install pytorch-scatter -c pyg
conda install -c conda-forge threadpoolctl
```

## Further Objectives and Implementation

### Checkpoint Loading
As of writing, there is no implementation for loading the checkpoints to resume training. I have implemented a proccess to save model checkpoints, but not yet a way to utilize them. It should not be difficult to do, as it would just need to involve the creation of new shell and python scripts that will load in the checkpoints and continue training.

### Point Classification to Event Classification
Currently, the model is for point classification. I believe that the MinkUNet Model needs to be modified as well as the upsampling layers should actually be removed, but this is not high priority. The next main project will be the implementation of event-wise classification for track counting. The current architecture is for point classification, so there will need to be changes with how the data and model architecture is structured. Essentially, it will be a completely new separate data proccessing and training pipeline with new data shape and architecture for explicitly track counting tasks.

## Original TorchSparse MIT GitHub and Links

### [website](http://torchsparse.mit.edu/) | [paper](https://arxiv.org/abs/2204.10319) | [presentation](https://www.youtube.com/watch?v=IIh4EwmcLUs) | [documents](http://torchsparse-docs.github.io/) | [pypi server](http://pypi.hanlab.ai/simple/torchsparse)

## Usage
[![DOI](https://zenodo.org/badge/736441968.svg)](https://zenodo.org/doi/10.5281/zenodo.10436993)

If you decide to use code in this repository, please cite as follows:

```
Heung, I. (2023). AT-TPC TorchSparse: 3D Sparse Convolution for AT-TPC Data (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.10436993
```

## Acknowledgement
Training Pipeline and Implementation based on code by MIT TorchSparse team. Please reference their repositories for more information:
- [TorchSparse](https://github.com/mit-han-lab/torchsparse/tree/master)
- [SPVNAS](https://github.com/mit-han-lab/spvnas/tree/dev/torchsparsepp_backend)

MinkUNet Model is adapted from [this code](https://github.com/mit-han-lab/spvnas/blob/dev/torchsparsepp_backend/core/models/semantic_kitti/minkunet.py) within the SPVNAS Repo

I would like to thank Dr. Kuchera, Dr. Ramanujan, and fellow ALPhA members for assiting and advising me throughout the project.
