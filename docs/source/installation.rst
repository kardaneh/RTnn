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

   This step might need to be ignored in the case that conflict of the environment modules arises. In that case, you can install the dependencies via pip (see next section).

   .. code-block:: bash

      module load profile/deeplrn
      module load cineca-ai

2. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/rtnn.git
      cd rtnn

3. **Create virtual environment**:

   .. code-block:: bash

      uv venv --python 3.11
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

If you're not on Leonardo, you can install all dependencies via pip (remember to uncomment the dependencies in ``pyproject.toml``):

.. code-block:: bash

   uv pip install -e .
   # This will install all dependencies including PyTorch

Development Installation
------------------------

For development, install with dev dependencies:

.. code-block:: bash

   uv pip install -e ".[dev]"
   pre-commit install

Building Documentation
----------------------

To build the HTML documentation locally:

1. **Install documentation dependencies**:

   .. code-block:: bash

      uv pip install -e ".[docs]"

   This installs Sphinx, the Read the Docs theme, and other required extensions.

2. **Navigate to the docs directory**:

   .. code-block:: bash

      cd docs

3. **Build the HTML documentation**:

   .. code-block:: bash

      make clean
      make html

   The HTML files will be generated in ``docs/build/html/``.

4. **View the documentation**:

   .. code-block:: bash

      # Open with your browser
      firefox build/html/index.html

      # Or serve with Python
      python -m http.server --directory build/html 8000


Dependencies
------------

The following packages are provided by the cineca-ai module:

- PyTorch 2.0.0a0 with CUDA 12.1
- NumPy, SciPy, Pandas, Matplotlib
- scikit-learn, seaborn, rich, tqdm, tensorboard

Only these need to be installed:

- xarray
- mpltex
