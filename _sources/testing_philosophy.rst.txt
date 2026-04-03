Testing Philosophy
==================

Unit Tests
----------

RTnn uses Python's built-in `unittest` framework.

Running tests:

.. code-block:: bash

   python -m unittest discover tests -v

   # Or using the test runner
   python tests/test_runner.py

Test Structure
--------------

- `test_rnn.py`: Tests for RNN models
- `test_fcn.py`: Tests for FCN models
- `test_transformer.py`: Tests for Transformer models

Test Data
---------

Tests use dynamically generated NetCDF files that mimic real data structure:
- Naming: `rtnetcdf_XXX_YYYY.nc`
- Multi-year and multi-processor support
- Realistic data dimensions

Coverage
--------

Tests cover:
- Model initialization
- Forward pass shapes
- Gradient flow
- Model serialization
- Device transfer
- Error handling
