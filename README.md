# Local connectivity mode
A stochastic graph model that captures non-random trends in local connectivity of neuronal circuitry.
With a proof-of-concept extension to long-range connectivity.

Primarily intended as a model of specifically cortical connectivity, but with flexibility that should make it possible candidate for other systems as well.

## Repository structure
This repository comprises two parts: A (small) python package that contains the functions to configure, build and compare instances of th graph model, and a number of jupyter notebooks where we use the package to characterize the properties of the model, compare manually optimized instances to references and generate the figures of the accompanyig paper.

# How to use the code
## Requirements
Required packages are listed in src/requirements.txt.
All packages are available on  pypi.
Note that at the moment the required connectome_analysis package is not available for python3.9, hence we suggest python3.8.

## Installation
Currently, we do not provide an installer for the python package. Instead, one is expected to clone this repository and run the provided notebooks. In the notebooks, the package is imported from local paths. 

We will provide a more easily usable solution in the future.

## Structure
The functionality is split between four python files:
 - **nngraph.py**: Contains functionality for the generation of a random geometric graph, including some modifications that are described in the manuscript.
 - **instance.py**: Contains functionality for the generation of a stochastic spread graph, as described in the manuscript.
 - **util.py**: Contains functionality for loading a reference connectome, or generating point clouds to build a graph on.
 - **test.py**: Contains functionality for comparing a generated graph to a reference and generate plots to that effect.

## Configuration
All parameterization is performed in .json files. Json files that have been used to generate the figures in the manuscript can be found under /configs.

Briefly, the files have four sections:
 - **make_points**: Parameterizes the generation of a point cloud that the graph(s) are generated on. This comes in two variants: Loading the soma positions from a reference connectome, or generating a random point cloud.
 - **nngraph**: Parameterizes the setup of the random geometric graph.
 - **per_class_bias**: Additional parameterization of the random geometric graph.
 - **instance**: Parameterizes the generation of the stochastic spread graph.

 The entries of "per_class_bias" specify a node property of a reference connectome (such as neuron type). The strengths of pathways with respect to that property are then approximately reproduced by calculating "per node biases" as described in the accompanying manuscript.

 For the other three sections, entries directly correspond to kwargs of python functions in the files listed above. Refer to the function docstrings for details.

# Notebooks
 - **Characterization.ipynb**: Performs sweeps over the paramters of the graph model for the purpose of characterization. Produces panels for *Fig 1C, D* of the manuscript.
 - **Match and compare to microns.ipynb**: Creates instances of the graph model for manually optimized parameters and compares the results to a reference. Creates the panels of *Fig 2*, *Fig 3* and *Fig S2*.
 - **Microns nearest neighbor effect**: Creates *Fig 1B* and *Fig S3*.
 - **Long range extension.ipynb**: Sketches an extension of the graph model to long-range connectivity. Generates the panels of *Fig 4*.
 - **Characterization_recursive.ipynb**: Characterizes a stochastic spread graph on a stochastic spread graph. Not used in the manuscript.


