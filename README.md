# Introduction

Contained within this repository you will find models and processing scripts relating to DSFU-Net, a generative deep learning method developed to take a plane of diffuse scattering input and factorise it into the contributions from molecular form factor and chemical short-range order. It is designed for single crystal systems exhibiting binary substitutional disorder, with one disordered site per unit cell. While it may be useful for other systems, the interpretation of the separated components is not clear outside of these assumptions. For more information on the limitations to check whether DSFU-Net is appropiate for your data, please refer to the paper given below.

We describe the various scripts and models present within this repo below as well as a demo on one of the validation examples such that you can see how it works with a working example. Please note this repo does not contain the training data as it is 100s of GB, but it does include the means to recreate it. 

# Citation

If this work proves useful to you, please cite us with:

Fuller and Rudden, DSFU-Net for unravelling diffuse scattering components, https://github.com/dclw29/DSFU-Net

Paper reference is TBC

# Installation

DSFU-Net is written entirely in Python and requires various packages, including PyTorch, to run. We describe below the necessary packages and how you might install it in a safe manner on your system.

DSFU-Net should work with the most updated version of Python (as of July 2023), but we will assume you are using Python 3.10 to keep it compatible with the work described in our paper. We recommend creating a safe conda environment (https://docs.conda.io/projects/conda/en/latest/index.html) to work in. For example,

```
conda create --name py310 python=3.10
```

Creates a Python 3.10 environment within GNU/Linux and other unix-based operating systems. This can then be activated:

```
conda activate py310
```

You can deactivate the environment with:

```
conda deactivate
```

And subsequently delete the environment if you desire:

```
conda remove --name py310 --all
```

You will need to install the following Python packages to run DSFU-Net:

- numpy
- matplotlib
- PIL
- argparse
- math
- itertools
- PyTorch

If you want to wish to recreate the dataset from scratch, you will need the following packages:

- geomloss
- pickle
- pandas
- Scipy

Most of the above will be automatically installed with Python 3.10, with the additional packages installable with:

```
pip install pillow matplotlib
pip install torch torchvision torchaudio
```

With the PyTorch pip installation, we are assuming you have a GPU onboard that PyTorch can recognise and harness when running your data through the network. A GPU is not necessary to run DSFU-Net, but significantly speeds up any runtimes, and any big data analysis/automated beamline workflow will need a GPU installed. You can install a CPU compatible version of PyTorch with:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

In general, PyTorch installations can be quite fickle, hence the recommendation for a conda environment. More information is available at https://pytorch.org/

# Running your sample
To prepare your data, you will need to reconstruct your reciprocal lattice planes on a 256x256 pixel grid and remove any Bragg peaks. For the best results, you could also apply symmetry averaging and subtract the background. If you want to do quantitative refinements of short-range order parameters, you need to be quite precise with the Bragg removal. If using a punch-and-fill method, its recommended to replace the Bragg intensities with noise instead of a constant value, for example. You will need to save your reconstructed plane as a numpy array. 

The key script for running your sample is given by main/pipeline.py. This can take as input a set of arguments, which you can interrogate with:

```
python pipeline.py --help
```
 
Crucially, there are two main modes to running this script using either the --filename or --folder flags. 

The first, when pointed to a specific filename, will read said filename (which MUST be a numpy array of size 256x256) and run DSFU-Net on that filename. If you instead specifiy a folder, then all files that are numpy arrays within that folder will be read and interpreted.

We provide a demo to test the network. This is available in /demo/. Simply run 

```
./run.sh
```

to automatically run the test0.npy example in the INPUT/ folder. This will place all form factor (FF) and short-range order (SRO) outputs in the OUTPUT/ folder. Feel free to use this run script as the basis for your own data, or run the other provided examples 1-3. This demo can also be used to test your installation of DSFU-Net.

# Creating your own dataset / retraining the network

We also provide the means to generate your own training data using either the precompiled parameters we discuss in the paper or your own, and then retrain the network. This could be useful, for example, if you want to change the shape of data to a different resolution (although note that interpolation to 256x256 is likely a simpler idea).

The bulk of scripts needed for dataset generation are in /dataset_generation/, though the prepare_lmdb_input.py script in /main/ is also needed. The main scripts of interest are the GenerateTrainingData, which differ as follows:

(1) GenerateTrainingData - the main dataset generation method (see paper for details) which pulls molecules from the meta_library files and uses the SRO parameters in the CorrelationGeneration folder to generate data. The lists of SRO parameters were obtained from Monte Carlo simulations.

(2) Tetragonal_DampedOscillator - as above, but uses a damped oscillator function to produce the SRO parameters instead of the Monte Carlo approach, on a tetragonal lattice

(3) Hexagonal_DampedOscillator - as in (2) but on a hexagonal lattice

The workflow discussed in the Methods of our paper involving this generation is all automated. The only changes you should have to make are:

- Update the folder locations, i.e. where do you want the training data saved? We have designated areas you will need to update in the Compile_Dataset/GenerateTrainingData scripts with a "/path/to/" root string.
- Choose whether to use the Tetragonal/Hexagonal or base GenerateTrainingData scripts, which will need corresponding edits of the Compile_Dataset script (see comments within for more details). 

You will also need to unzip the Artefacts folder and put the correct location where specified in the Compile_Dataset script.

Finally, you can also retrain the network, or design your own tweaks by editing the models script in /main/ if you desire. We include the self-attention and dynamic Pix2PixGANs in the models script discussed in the supplemental information should you want to use or play with those. Please note that the self-attention Pix2PixGAN is not published, but rather a product of our own experimentation. The Dynamic Pix2PixGAN, while based on previous published work (https://arxiv.org/abs/2211.08570), is our own guess from the limited information given in the paper as no code or GitHub was given. 

We provide a train.sh bash script to run the pix2pix-gan given in the /main/ folder, but many more arguments than those used in train are available with their own default settings. You can find these with:

```
python dsfu-net.py --help
```

# Contact

For questions on the generation of the training data, please contact Chloe at chloe.fuller@esrf.fr

For questions or issues with the network, please submit a ticket above, or contact Lucas at: lucas.rudden@epfl.ch

