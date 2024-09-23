# Latent_Feature_Grid_Compression

[**Project**]([https://bussler.github.io/latent-feature-grid-encoding/](https://bussler.github.io/#/latentfeaturegridcompression)) | [**Master's Thesis**](Master_Thesis_Training_Methods_for_Memory_efficient_Volume_Scene_Representation_Networks_Maarten_Bussler.pdf)

Project for my master's thesis to research possibilities of compressing Scene Representation Networks based on latent feature grids with network pruning algorithms.
The aim is to use the pruning algorithms for identifying the network parameters most important to the reconstruction quality and eliminating the less significant ones.
This enables the network to learn a good tradeoff between compression ratio and reconstruction quality.

The network is based on [FV-SRN](https://github.com/shamanDevel/fV-SRN). Besides a binary masking pruning, the pruning algorithms of [Smallify](https://github.com/mitdbg/fastdeepnets) and [Variational Dropout](https://arxiv.org/pdf/1506.02557.pdf) are implemented.

## Quick Start

### Requirements
Basic functionalities, such as training and quantization of the network with 3D numpy arrays as input, as well as writing of the results as .vti files can be enabled by installing with pip (`Env.txt`) or conda (`Env.yml`).
The resulting .vti files can be visualized with [ParaView](https://www.paraview.org/).

Optionally: [Pyrenderer](https://github.com/shamanDevel/fV-SRN) can be used as a visualization tool and to feed CVol Data into the network.
Follow the instructions under https://github.com/shamanDevel/fV-SRN for installation.

### Data
Datasets and corresponding config files for all experiments can be found in `datasets/` and `experiment-config-files/`.

### Train and run the network:
1. Install the requirements from Env.txt (pip) or Env.yml (conda).
2. Generate a config-file for the experiment or use one under `experiment-config-files/`. Descriptions for the different parameters can be generated with `python Feature_Grid_Training.py --help`.
3. Use `python Feature_Grid_Training.py --config <Path-To-Config-File>` to start training.
4. During training, [Tensorboard](https://mlflow.org/docs/latest/quickstart.html) tracks the experiment under `runs/`.
After training, a checkpoint to the trained model, as well as the config-file and basic information about the training are logged to `experiments/<expname>/`. 
The checkpoint is generated in two ways: First, as a torch .pth (`model.pth`) for easy reading with other torch implementations. Secondly, stored efficiently as a binary representation, where the pruned parameters are removed (`binary_model_file` and `binary_model_file_mask.bnr`).
Furthermore a .vti file for the ground-truth and model-predicted volume will be generated.
5. A generated model can be inferred again explicitly with `python Feature_Grid_Inference.py --config_path <Path-To-Config-File> --reconstruct <binary> | <checkpoint>`.
The paths to the binary masks and torch checkpoints are stored in the model config file, and the reconstruction source can be specified to reconstruct from the efficient binary representation with 'binary' or from the torch checkpoint with 'checkpoint'.

### Perform Hyperparameter Search:
In order to find the best hyperparameter for each network type and dataset, the [AX MULTI-OBJECTIVE NAS](https://ax.dev/) Algorithm is provided.
To run hyperparameter search, use `jupyter notebook` to start either the 'Multiobjective-NAS' jupyter notebook.
In the first cell, define the config file of the experiment, then execute the subsequent cells to start the scheduler and visualize the results.
The Search-Space for each experiment can be configured in `Multi_Objective_NAS.py`.

## Project Structure
- Parsing of arguments, as well as the entry points to training and inference are implemented in `Feature_Grid_Training.py` and `Feature_Grid_Inference.py`.
- The initialization of the network, as well as training is implemented in `training/training.py`.
- Model utilities, such as network setup and storage are implemented in `model/model_utils.py`.
- The basic model architecture can be found in `model/Feature_Grid_Model.py` and `model/Feature_Embedding.py`.
- The pruning algorithms are implemented in `model/Smallify_Dropout.py`, `model/Straight_Through_Dropout.py` and `model/Variational_Dropout_Layer.py`.
- Data input is handled in the classes in `data/`

## Results
All experiments are performed on the `datasets\mhd1024.h5`(255, 255, 255) and `datasets\test_vol.npy`(150,150,150) datasets.
The fv-SRN, as well as Neurcomp are able to outperform state of the art compression algorithms, like TTHRESH.

<img src="plots\baseline_comparisons.JPG" alt="example" width="1000" >

Pruning works best when performed on large, open networks, where singular nodes have relatively low influence on the reconstruction quality.
Pruning is performed on the latent feature grid of the fV-SRN and is able to significantly improve the reconstruction quality of the network by up to 5 PSNR points.

<img src="plots\baseline_prunings.JPG" alt="example" width="1000" >

The wavelet transformation is able to further enhance the effectiveness of the pruning algorithms.
This is because most of the latent feature information of the fV-SRN feature grid are encoded into just a few wavelet coefficients, enabling the pruning algorithms to easily distinguish between important and unimportant parameters.

<img src="plots\wavelet_gain.JPG" alt="example" width="1000" >

For a more extensive review, see the pdf of the [Master's thesis](Master_Thesis_Training_Methods_for_Memory_efficient_Volume_Scene_Representation_Networks_Maarten_Bussler.pdf).
