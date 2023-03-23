# Latent_Feature_Grid_Compression

[**FV-SRN**](https://github.com/shamanDevel/fV-SRN) | [**Smallify**](https://github.com/mitdbg/fastdeepnets) | [**Variational Dropout**](https://arxiv.org/pdf/1506.02557.pdf)

Project for my master's thesis to research possibilities of compressing networks based on latent feature grids.

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
4. During training, [Tensorboard](https://mlflow.org/docs/latest/quickstart.html) tracks the experiment under `runs/`. A checkpoint to the trained model, as well as the config-file and basic information about the training are logged to `experiments/<expname>/`. Also a .vti file for the ground-truth and model-predicted volume will be generated.
5. Explicit Inference: TODO

### Perform Hyperparameter Search:
In order to find the best hyperparameter for each network type and dataset, the [AX MULTI-OBJECTIVE NAS](https://ax.dev/) Algorithm is provided.
To run hyperparameter search, use `jupyter notebook` to start either the 'Multiobjective-NAS' jupyter notebook.
In the first cell, define the config file of the experiment, then execute the subsequent cells to start the scheduler and visualize the results.
The Search-Space for each experiment can be configured in `Multi_Objective_NAS.py`.

### Encode and decode the network with quantization:
TODO

## Project Structure
- Parsing of Arguments, as well as the entry points to training in `Feature_Grid_Training.py`.
- The initialization of the network, as well as training is implemented in `training/training.py`.
- The basic model architecture can be found in `model/NeurcompModel.py` and `model/SirenLayer.py`.
- The Pruning algorithms are implemented in `model/Smallify_Dropout.py`, `model/Straight_Through_Dropout.py` and `model/Variational_Dropout_Layer.py`.
- Data input is handled in the classes in `data/`

## Results
TODO