RTnn
====

RTnn (Radiative Transfer Neural Networks) is a PyTorch-based framework designed to
emulate radiative transfer processes in climate models, with a primary focus on
Land Surface Models (LSM). It provides efficient neural network surrogates that
can replace expensive physical radiative transfer calculations while maintaining
accuracy.

**GitHub Repository:** https://github.com/kardaneh/RTNN

**Key Features:**

- **Multiple neural architectures**: LSTM, GRU, Transformer, and FCN
- **Climate data support**: Native NetCDF4 handling with multi-year and multi-processor data
- **GPU acceleration**: CUDA support with multi-GPU training capabilities
- **Comprehensive evaluation**: Built-in metrics (NMAE, NMSE, R²) and visualization tools
- **Flexible preprocessing**: Multiple normalization schemes (minmax, standard, robust, log1p, sqrt)
- **Command-line interface**: Easy training and inference without coding

**Applications:**

- Emulating canopy radiative transfer in vegetation models
- Accelerating climate model simulations
- Data assimilation and uncertainty quantification
- Sensitivity analysis of radiative transfer parameters

**Performance:**

- Up to YYYx faster than physical RT models
- Minimal accuracy loss (typically >0.95 R²)
- Scalable to large datasets with distributed training

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
   data_handling

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
