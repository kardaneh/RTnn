Overview
========

RTnn (Radiative Transfer Neural Networks) is a PyTorch-based framework for training
neural networks to model radiative transfer processes in climate science, particularly
for Land Surface Models (LSM).

What is RTnn?
-------------

RTnn provides a flexible and efficient framework for:

- **Emulating radiative transfer**: Replace expensive physical RT models with fast neural networks
- **Data preprocessing**: Handle large climate datasets with multiple dimensions
- **Multiple architectures**: Support for LSTM, GRU, Transformer, and FCN models
- **GPU acceleration**: Leverage CUDA for fast training and inference
- **Comprehensive evaluation**: Built-in metrics and visualization tools

Problem Statement
-----------------

Radiative transfer calculations are among the most computationally expensive components
in climate models. These calculations determine how solar radiation interacts with:

- **Vegetation canopies** (absorption, reflection, transmission)
- **Atmospheric layers** (scattering, absorption, emission)
- **Surface properties** (albedo, emissivity)

Traditional physical models require solving complex radiative transfer equations
(e.g., two-stream approximation, discrete ordinates method) at each grid point and
time step, making them a significant computational bottleneck.

Solution Approach
-----------------

RTnn addresses this challenge by training neural networks to learn the input-output
mapping of radiative transfer processes:

**Input variables:**

- Solar zenith angle (coszang)
- Leaf area index (LAI) - collimated and isotropic
- Leaf single scattering albedo (SSA) and phase function asymmetry (PSD)
- Surface reflectance (rs_surface_emu)

**Output variables:**

- Collimated and isotropic albedo
- Collimated and isotropic transmittance
- Absorption rates (channels 1-2 and 3-4)

Architecture Overview
---------------------

.. code-block:: text

    Input Data (NetCDF) → DataPreprocessor → DataLoader → Model → Output
         ↓                      ↓              ↓          ↓         ↓
    rtnetcdf_XXX_YYYY.nc   Normalization   Batching    Forward   Predictions
                          Variable groups   Shuffle     Pass     Unnormalized
                         Spatial/temporal                          Results
                              batching

Key Components
--------------

1. **DataPreprocessor**: Handles loading and preprocessing of NetCDF files with
   multi-year and multi-processor data

2. **Model Architectures**: Multiple neural network options including LSTM, GRU,
   Transformer, and FCN

3. **Evaluation Framework**: Comprehensive metrics and loss functions for
   radiative transfer applications

4. **Command Line Interface**: Easy training and inference without coding

Performance Highlights
----------------------

- **Speed**: Up to YYYx faster than physical RT models
- **Accuracy**: R² > 0.95 across all output variables
- **Scalability**: Efficient on multi-GPU and distributed systems
- **Data efficiency**: Trained on ~5 years of data, validated on independent years

Next Steps
----------

- :doc:`installation` - Install RTnn on your system
- :doc:`quickstart` - First steps with RTnn
- :doc:`neural_architectures` - Learn about available model architectures
