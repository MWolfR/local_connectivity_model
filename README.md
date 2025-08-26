# Local connectivity model
A stochastic graph model that captures non-random trends in local connectivity of neuronal circuitry.
With a proof-of-concept extension to long-range connectivity.
The model defines a way to build a graph on top of the structure of another graph. We call this a *stochastic spread graph*. A stochastic spread graph on top of a random geometric graph we call a stochastic geometric spread graph, or SGSG.

SGSG is primarily intended as a model of specifically cortical connectivity, but with flexibility that should make it possible candidate for other systems as well.

## Repository structure
This repository comprises two parts: A (small) python package that contains the functions to configure, build and compare instances of th graph model, and a number of jupyter notebooks where we use the package to characterize the properties of the model, compare manually optimized instances to references and generate the figures of the accompanyig paper.

# How to use the code
## Requirements
Required packages are listed in src/requirements.txt.
All packages are available on  pypi.
Note that at the moment the required connectome_analysis package is not available for python3.13, hence we suggest python3.12.

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
We use parameterization that is written into .json files. Json files that have been used to generate manuscript figures can be found under /configs. They require reference connectomes that contain, e.g., the connectivity of the microns data. Instructions on where to find them are *inside the config files*! Reference connectomes are in an hdf5-based formated used by the Connectome-Utilities python package (listed on pypi).

Briefly, the files have four sections:
 - **make_points**: Parameterizes the generation of a point cloud that the graph(s) are generated on. This comes in two variants: Loading the soma positions from a reference connectome, or generating a random point cloud.
 - **nngraph**: Parameterizes the setup of the random geometric graph.
 - **per_class_bias**: Additional parameterization of the random geometric graph.
 - **instance**: Parameterizes the generation of the stochastic spread graph.

 The entries of "per_class_bias" specify a node property of a reference connectome (such as neuron type). The strengths of pathways with respect to that property are then approximately reproduced by calculating "per node biases" as described in the accompanying manuscript.

 For the other three sections, entries directly correspond to kwargs of python functions in the files listed above. Refer to the function docstrings for details.

 ### Simply getting started
 "Configuration file? That is all too complicated for me. Just tell me how to build S_q(G_{d,p}(P)), as you define it in the manuscript".

> import numpy
> 
> from pnagm.nngraph import cand2_point_nn_matrix
> 
> from pnagm.instance import build_instance
>
> 
> P = numpy.random.rand(1000, 3) * 50
>
> p = 0.2  # parameter as defined in the manuscript
>
> d = 10  # as in the manuscript
>
> q = 2.5  # as in the manuscript
>
> 
> G = cand2_point_nn_matrix(P, dist_neighbors=d, p_pick=p)  # Builds the random geometric graph
>
> S, _, _ = build_instance(P, G, step_tgt=q)  # Build stochastic spread graph on it

# Notebooks

## Core notebooks analyzing SGSGs
 - **Characterization.ipynb**: Performs sweeps over the paramters of the graph model for the purpose of characterization. Produces panels for *Fig 1C, D* of the manuscript.
 - **Match and compare to microns.ipynb**: Creates instances of the graph model for manually optimized parameters in a configuration file and compares the results to a reference. Also compares to various other models, such as distance-dependent.
 - **Long range extension.ipynb**: Sketches an extension of the graph model to long-range connectivity.
 - **Characterization_recursive.ipynb**: Characterizes a stochastic spread graph on a stochastic spread graph. Not used in the manuscript.


 ## Notebooks not directly related to SGSGs
 These notebooks are related to explaining the motivation for the SGSG model, or to comparisons with strong controls.
  - **Nearest neighbor 1d prob increase**: Calculates the increase in connection probability when the nearest neighbor of a neuron is confirmed to be connected.
  - **Nearest neighbor 2d prob increase**: Similar to the previous entry, but instead of connection probability against distance in one dimension, it considers two dimensions (horizontal and vertical offset).
  - **Conn prob correlations between spatial bins**: Calculates and displays Pearson correlations of connection probabilities between spatial bins over neurons. 
  - **Fit other models to microns**: This notebooks generates various forms of preferential attachment models of connectivity, fit to the MICrONS reference data. It then generates instances of the model to be used in other notebooks!
  - **Microns mean l5 connection prob**: Simply generates an estimate of the connection probability within 100 um of L5 PCs from the MICrONS data.
  - **Draw explanatory cartoons**: Draws cartoons that explain some of the concepts of the graph models we use. Not very interesting.
  - **Microns nearest neighbor effect**: An earlier version of "Nearest neighbor 1d prob increase".

# Generating paper figures
Code from this repository is used in two papers. Below, we list which panels of which papers are generated by which notebooks. Note that the generation of the figures requires additional data files (hdf5 format) that hold the relevant connectome data. Read the instruction in this file and the notebooks carefully to find out how to obtain them!

## An algorithm to model the non-random connectome of cortical circuitry
biorXiv; doi: 10.1101/2025.05.22.655546

  - Figure 1A: **Draw explanatory cartoons**
  - Figure 1B: **Microns nearest neighbor effect**
  - Figure 1C, D: **Characterization**
  - Figure 2: **Match and compare to microns** with config L45E_microns_yscale_experimental_v1p5.json
  - Figure 3: **Match and compare to microns** with config L45E_SSCX_yscale_experimental_v1p5.json
  - Figure 4: **Long range extension** and **Draw explanatory cartoons**
  - Figure S2: **Match and compare to microns** with config L23E_microns_yscale_experimental_v1p5.json
  - Figure S3: **Microns nearest neighbor effect**

## Neuron morphological physicality and variability define the non-random structure of connectivity
biorXiv; doi: 10.1101/2025.08.21.671478

  - Figure 1A: Reproduced from a previous paper. Not in this repository
  - Figure 1B, C: Manually drawn cartoon
  - Figure 1D1: **Nearest neighbor 1d prob increase** with the *data* connectome from 10.5281/zenodo.16744240 and 10.5281/zenodo.16744766
  - Figure 1D2: **Nearest neighbor 1d prob increase** with the *control* connectome from 10.5281/zenodo.16744240 and https://doi.org/10.5281/zenodo.16744766
  - Figure 2A: **Nearest neighbor 2d prob increase** with the *data* connectome from 10.5281/zenodo.16744240
  - Figure 2B: **Nearest neighbor 2d prob increase** with the *data* connectome from 10.5281/zenodo.16744766
  - Figure 2C: **Nearest neighbor 2d prob increase** with the *data* connectome from 10.5281/zenodo.16744240 and a configuration model control of the same connectome (use the connectome-analysis package to generate)
  - Figure 2D: **Nearest neighbor 2d prob increase** with the *data* connectome from 10.5281/zenodo.16744766 and the *control* connectome from 10.5281/zenodo.16744766
  - Figure 3: **Conn prob correlations between spatial bins** with the *data* connectome from 10.5281/zenodo.16744240 and a configuration model control of the same connectome (use the connectome-analysis package to generate)
  - Figure 4A: **Draw explanatory cartoons**
  - Figure 4B-E: **Match and compare to microns** with config L45E_microns_yscale_experimental_v1p5.json
  - Figure 5: **Long range extension** and **Draw explanatory cartoons**
  
