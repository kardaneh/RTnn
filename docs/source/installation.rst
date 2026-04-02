Installation
============

Requirements
------------

- Python 3.8 or higher
- PyTorch 2.0+ (provided by cineca-ai module)
- CUDA-capable GPU (optional, for GPU acceleration)

Installation on Leonardo (CINECA)
---------------------------------

1. **Load required modules**:

   .. code-block:: bash

      module load profile/deeplrn
      module load cineca-ai

2. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/rtnn.git
      cd rtnn

3. **Create virtual environment**:

   .. code-block:: bash

      uv venv
      source .venv/bin/activate

4. **Install missing dependencies**:

   .. code-block:: bash

      uv pip install xarray mpltex

5. **Install the package**:

   .. code-block:: bash

      uv pip install -e .

6. **Verify installation**:

   .. code-block:: bash

      python -c "import rtnn; print(rtnn.__version__)"
      rtnn --version

Installation on other systems
-----------------------------

If you're not on Leonardo, you can install all dependencies via pip:

.. code-block:: bash

   uv pip install -e .
   # This will install all dependencies including PyTorch

Development Installation
------------------------

For development, install with dev dependencies:

.. code-block:: bash

   uv pip install -e ".[dev]"
   pre-commit install

Dependencies
------------

The following packages are provided by the cineca-ai module:
- PyTorch 2.0.0a0 with CUDA 12.1
- NumPy, SciPy, Pandas, Matplotlib
- scikit-learn, seaborn, rich, tqdm, tensorboard

Only these need to be installed:
- xarray
- mpltex
