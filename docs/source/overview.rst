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
- **Multiple architectures**: Support for LSTM, GRU, Transformer, FCN, and UNet models
- **GPU acceleration**: Leverage CUDA for fast training and inference
- **Comprehensive evaluation**: Built-in metrics and visualization tools

Key Features
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Description
   * - Model Architectures
     - LSTM, GRU, Transformer, FCN, UNet1D
   * - Data Handling
     - NetCDF4 support, multi-year data, spatial/temporal batching
   * - Normalization
     - Multiple schemes: minmax, standard, robust, log1p, sqrt
   * - Loss Functions
     - MSE, MAE, NMSE, NMAE, Huber, LogCosh, Weighted MSE
   * - Visualization
     - Training curves, prediction vs target plots, metrics dashboards
   * - GPU Support
     - Multi-GPU training with CUDA acceleration

Applications
------------

RTnn is designed for:

1. **Land Surface Models (LSM)**: Emulate radiative transfer in vegetation canopies
2. **Atmospheric Science**: Model radiation in atmospheric columns
3. **Climate Downscaling**: Learn relationships between coarse and fine resolutions
4. **Data Assimilation**: Fast forward operators for ensemble methods

Citation
--------

If you use RTnn in your research, please cite:

.. code-block:: bibtex

   @software{ardaneh2026rtnn,
       author = {Kazem Ardaneh},
       title = {RTnn: Radiative Transfer Neural Networks for Climate Science},
       year = {2026},
       publisher = {GitHub},
       url = {https://github.com/kardaneh/rtnn}
   }
