Testing Philosophy
===================

RTnn uses Python's built-in `unittest` framework for comprehensive testing.

Running Tests
-------------

Run all tests:

.. code-block:: bash

   python -m unittest discover tests -v

Run specific test modules:

.. code-block:: bash

   python -m unittest tests.test_rnn -v
   python -m unittest tests.test_fcn -v
   python -m unittest tests.test_transformer -v
   python -m unittest tests.test_dataset -v

Run with test runner (rich output):

.. code-block:: bash

   python tests/test_runner.py

Test Coverage
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Test Module
     - Coverage
   * - ``test_rnn.py``
     - LSTM and GRU models (initialization, forward pass, gradients)
   * - ``test_fcn.py``
     - FCN model (forward pass, dimension expansion)
   * - ``test_transformer.py``
     - Transformer encoder (attention, blocks)
   * - ``test_dataset.py``
     - DataPreprocessor (loading, normalization, indexing)
   * - ``test_evaluater.py``
     - Loss functions and metrics
   * - ``test_model_loader.py``
     - Model factory function

Test Data Generation
--------------------

Tests use dynamically generated NetCDF files that mimic real data:

- File naming: ``rtnetcdf_XXX_YYYY.nc``
- Multi-year and multi-processor support
- Realistic dimension ordering
- All required variables present

Example test structure
----------------------

.. code-block:: python

   class TestRNN_LSTM(unittest.TestCase):
       def setUp(self):
           self.model = RNN_LSTM(
               feature_channel=6,
               output_channel=4,
               hidden_size=64,
               num_layers=2
           )

       def test_forward_shape(self):
           x = torch.randn(32, 6, 10)
           y = self.model(x)
           self.assertEqual(y.shape, (32, 4, 10))

Running a Single Test
---------------------

.. code-block:: bash

   python -m unittest tests.test_rnn.TestRNN_LSTM.test_forward_shape -v

Continuous Integration
----------------------

GitHub Actions automatically runs tests on:

- Push to ``master`` branch
- Pull requests to ``master``
- Manual workflow dispatch
