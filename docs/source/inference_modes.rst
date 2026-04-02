Inference Modes
===============

Running inference with trained models.

Command Line
------------

.. code-block:: bash

   rtnn --inference True --load_model True --load_checkpoint_name model.pth

Python API
----------

.. code-block:: python

   from rtnn import load_model, DataPreprocessor

   # Load trained model
   model = load_model(args)
   checkpoint = torch.load('model.pth')
   model.load_state_dict(checkpoint['state_dict'])

   # Run inference
   model.eval()
   with torch.no_grad():
       predictions = model(input_data)
