# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtnn.model_loader import load_model
from rtnn.models.rnn import RNN_LSTM, RNN_GRU
from rtnn.models.Transformer import Encoder
from rtnn.models.fcn import FCN


class TestModelLoader(unittest.TestCase):
    """Unit tests for model_loader factory function."""

    def setUp(self):
        """Set up test fixtures with common parameters."""
        self.common_params = {
            "feature_channel": 6,
            "output_channel": 4,
            "seq_length": 10,
        }

    # ------------------------------------------------------------------------
    # LSTM Tests
    # ------------------------------------------------------------------------

    def test_load_lstm_model(self):
        """Test loading LSTM model."""
        args = argparse.Namespace(
            type="lstm",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            hidden_size=128,
            num_layers=3,
            seq_length=self.common_params["seq_length"],
        )

        model = load_model(args)

        self.assertIsInstance(model, RNN_LSTM)
        self.assertEqual(model.output_channel, args.output_channel)
        self.assertEqual(model.hidden_size, args.hidden_size)
        self.assertEqual(model.num_layers, args.num_layers)

    def test_load_lstm_with_different_params(self):
        """Test loading LSTM with different parameters."""
        args = argparse.Namespace(
            type="lstm",
            feature_channel=8,
            output_channel=6,
            hidden_size=256,
            num_layers=4,
            seq_length=20,
        )

        model = load_model(args)

        self.assertIsInstance(model, RNN_LSTM)
        self.assertEqual(model.output_channel, 6)
        self.assertEqual(model.hidden_size, 256)
        self.assertEqual(model.num_layers, 4)

    # ------------------------------------------------------------------------
    # GRU Tests
    # ------------------------------------------------------------------------

    def test_load_gru_model(self):
        """Test loading GRU model."""
        args = argparse.Namespace(
            type="gru",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            hidden_size=128,
            num_layers=3,
            seq_length=self.common_params["seq_length"],
        )

        model = load_model(args)

        self.assertIsInstance(model, RNN_GRU)
        self.assertEqual(model.output_channel, args.output_channel)
        self.assertEqual(model.hidden_size, args.hidden_size)
        self.assertEqual(model.num_layers, args.num_layers)

    def test_load_gru_with_different_params(self):
        """Test loading GRU with different parameters."""
        args = argparse.Namespace(
            type="gru",
            feature_channel=10,
            output_channel=8,
            hidden_size=512,
            num_layers=5,
            seq_length=30,
        )

        model = load_model(args)

        self.assertIsInstance(model, RNN_GRU)
        self.assertEqual(model.output_channel, 8)
        self.assertEqual(model.hidden_size, 512)
        self.assertEqual(model.num_layers, 5)

    # ------------------------------------------------------------------------
    # Transformer Tests
    # ------------------------------------------------------------------------

    def test_load_transformer_model(self):
        """Test loading Transformer model."""
        args = argparse.Namespace(
            type="transformer",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            embed_size=64,
            num_layers=2,
            nhead=4,
            forward_expansion=4,
            seq_length=self.common_params["seq_length"],
            dropout=0.1,
        )

        model = load_model(args)

        self.assertIsInstance(model, Encoder)
        self.assertEqual(model.embed_size, args.embed_size)
        self.assertEqual(model.seq_length, args.seq_length)

    def test_load_transformer_with_forward_expansion_none(self):
        """Test loading Transformer when forward_expansion is None."""
        args = argparse.Namespace(
            type="transformer",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            embed_size=64,
            num_layers=2,
            nhead=4,
            forward_expansion=None,  # Should default to 1
            seq_length=self.common_params["seq_length"],
            dropout=0.1,
        )

        model = load_model(args)

        self.assertIsInstance(model, Encoder)

    def test_load_transformer_with_different_params(self):
        """Test loading Transformer with different parameters."""
        args = argparse.Namespace(
            type="transformer",
            feature_channel=12,
            output_channel=8,
            embed_size=128,
            num_layers=4,
            nhead=8,
            forward_expansion=8,
            seq_length=20,
            dropout=0.2,
        )

        model = load_model(args)

        self.assertIsInstance(model, Encoder)
        self.assertEqual(model.embed_size, 128)
        self.assertEqual(model.seq_length, 20)

    # ------------------------------------------------------------------------
    # FCN Tests
    # ------------------------------------------------------------------------

    def test_load_fcn_model(self):
        """Test loading FCN model."""
        args = argparse.Namespace(
            type="fcn",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            num_layers=3,
            hidden_size=196,
            seq_length=self.common_params["seq_length"],
        )

        model = load_model(args)

        self.assertIsInstance(model, FCN)
        self.assertEqual(model.feature_channel, args.feature_channel)
        self.assertEqual(model.output_channel, args.output_channel)
        self.assertEqual(model.seq_length, args.seq_length)

    def test_load_fullyconnected_model(self):
        """Test loading FCN using 'fullyconnected' alias."""
        args = argparse.Namespace(
            type="fullyconnected",
            feature_channel=self.common_params["feature_channel"],
            output_channel=self.common_params["output_channel"],
            num_layers=3,
            hidden_size=196,
            seq_length=self.common_params["seq_length"],
        )

        model = load_model(args)

        self.assertIsInstance(model, FCN)

    def test_load_fcn_with_different_params(self):
        """Test loading FCN with different parameters."""
        args = argparse.Namespace(
            type="fcn",
            feature_channel=16,
            output_channel=10,
            num_layers=5,
            hidden_size=512,
            seq_length=50,
        )

        model = load_model(args)

        self.assertIsInstance(model, FCN)
        self.assertEqual(model.feature_channel, 16)
        self.assertEqual(model.output_channel, 10)
        self.assertEqual(model.seq_length, 50)
        self.assertEqual(model.dim_expand, 0)

    # ------------------------------------------------------------------------
    # Invalid Input Tests
    # ------------------------------------------------------------------------

    def test_invalid_model_type(self):
        """Test loading with invalid model type raises ValueError."""
        args = argparse.Namespace(
            type="invalid_model",
            feature_channel=6,
            output_channel=4,
            hidden_size=128,
            num_layers=3,
        )

        with self.assertRaises(ValueError) as context:
            load_model(args)

        self.assertIn(
            "Model type 'invalid_model' is not implemented", str(context.exception)
        )

    def test_case_insensitive_type(self):
        """Test that model type is case-insensitive."""
        args = argparse.Namespace(
            type="LSTM",  # Uppercase
            feature_channel=6,
            output_channel=4,
            hidden_size=128,
            num_layers=3,
            seq_length=10,
        )

        model = load_model(args)

        self.assertIsInstance(model, RNN_LSTM)

    def test_missing_required_args_lstm(self):
        """Test missing required arguments for LSTM."""
        args = argparse.Namespace(
            type="lstm",
            # Missing feature_channel, output_channel, etc.
        )

        with self.assertRaises(AttributeError):
            load_model(args)

    def test_missing_required_args_transformer(self):
        """Test missing required arguments for Transformer."""
        args = argparse.Namespace(
            type="transformer",
            feature_channel=6,
            # Missing embed_size, nhead, etc.
        )

        with self.assertRaises(AttributeError):
            load_model(args)

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------

    def test_loaded_model_forward_pass(self):
        """Test that loaded model can perform a forward pass."""

        args = argparse.Namespace(
            type="lstm",
            feature_channel=6,
            output_channel=4,
            hidden_size=64,
            num_layers=2,
            seq_length=10,
        )

        model = load_model(args)
        x = torch.randn(32, 6, 10)
        y = model(x)

        self.assertEqual(y.shape, (32, 4, 10))

    def test_loaded_transformer_forward_pass(self):
        """Test that loaded Transformer can perform a forward pass."""

        args = argparse.Namespace(
            type="transformer",
            feature_channel=6,
            output_channel=4,
            embed_size=64,
            num_layers=2,
            nhead=4,
            forward_expansion=4,
            seq_length=10,
            dropout=0.1,
        )

        model = load_model(args)
        x = torch.randn(32, 6, 10)
        y = model(x)

        self.assertEqual(y.shape, (32, 4, 10))

    def test_loaded_fcn_forward_pass(self):
        """Test that loaded FCN can perform a forward pass."""

        args = argparse.Namespace(
            type="fcn",
            feature_channel=6,
            output_channel=4,
            num_layers=3,
            hidden_size=196,
            seq_length=10,
        )

        model = load_model(args)
        x = torch.randn(32, 6, 10)
        y = model(x)

        self.assertEqual(y.shape, (32, 4, 10))

    # ------------------------------------------------------------------------
    # Parameter Count Tests
    # ------------------------------------------------------------------------

    def test_lstm_has_parameters(self):
        """Test that loaded LSTM model has trainable parameters."""
        args = argparse.Namespace(
            type="lstm",
            feature_channel=6,
            output_channel=4,
            hidden_size=128,
            num_layers=3,
            seq_length=10,
        )

        model = load_model(args)
        total_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(total_params, 0)

    def test_transformer_has_parameters(self):
        """Test that loaded Transformer model has trainable parameters."""
        args = argparse.Namespace(
            type="transformer",
            feature_channel=6,
            output_channel=4,
            embed_size=64,
            num_layers=2,
            nhead=4,
            forward_expansion=4,
            seq_length=10,
            dropout=0.1,
        )

        model = load_model(args)
        total_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(total_params, 0)

    def test_fcn_has_parameters(self):
        """Test that loaded FCN model has trainable parameters."""
        args = argparse.Namespace(
            type="fcn",
            feature_channel=6,
            output_channel=4,
            num_layers=3,
            hidden_size=196,
            seq_length=10,
        )

        model = load_model(args)
        total_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(total_params, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoader))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
