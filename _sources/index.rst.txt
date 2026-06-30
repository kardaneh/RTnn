RTnn
====

RTnn (Radiative Transfer Neural Networks) is a PyTorch-based framework for
emulating radiative transfer processes in climate models, with a primary
application to Land Surface Models (LSM). It provides neural network surrogates
that replace computationally expensive physical radiative transfer schemes
while preserving accuracy.

GitHub repository: https://github.com/kardaneh/RTNN

Key Features
------------

- **Multiple architectures**: LSTM, GRU, Transformer, FCN, MLP
- **Climate data support**: Native NetCDF4 handling with multi-year, multi-process workflows
- **GPU acceleration**: CUDA support with multi-GPU training
- **Comprehensive evaluation**: Built-in metrics (NMAE, NMSE, R²) and diagnostics/visualization tools
- **Flexible preprocessing**: Multiple normalization methods (minmax, standard, robust, log1p, sqrt)
- **Command-line interface**: Training and inference directly from CLI without code changes

Applications
------------

- Emulation of canopy radiative transfer in vegetation models
- Acceleration of climate model simulations
- Data assimilation and uncertainty quantification
- Sensitivity analysis of radiative transfer parameters

Performance
-----------

- Up to YYY× speed-up compared to physical radiative transfer models
- Minimal accuracy loss (typically R² > 0.9999)
- Scalable to large datasets with multi-GPU training

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   neural_architectures
   training_strategy
   inference_modes

.. toctree::
   :maxdepth: 2
   :caption: Benchmark

   lsm_benchmark
   atm_benchmark
   ref_trans_benchmark

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   project_structure
   testing_philosophy
   pre_push_workflow

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
