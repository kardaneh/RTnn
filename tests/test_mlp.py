# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtnn.models.mlp import MLP, MLPBlock, MLPResidual


class TestMLPBlock(unittest.TestCase):
    """Unit tests for MLPBlock class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.in_features = 128
        self.out_features = 64
        self.dropout = 0.1

    def test_initialization(self):
        """Test MLPBlock initialization."""
        block = MLPBlock(self.in_features, self.out_features, dropout=self.dropout)

        self.assertIsInstance(block.linear, torch.nn.Linear)
        self.assertEqual(block.linear.in_features, self.in_features)
        self.assertEqual(block.linear.out_features, self.out_features)

        self.assertIsInstance(block.bn, torch.nn.BatchNorm1d)
        self.assertEqual(block.bn.num_features, self.out_features)

        self.assertIsInstance(block.act, torch.nn.ReLU)
        self.assertIsInstance(block.dropout, torch.nn.Dropout)
        self.assertEqual(block.dropout.p, self.dropout)

    def test_initialization_with_layer_norm(self):
        """Test MLPBlock initialization with layer norm."""
        block = MLPBlock(
            self.in_features,
            self.out_features,
            use_batch_norm=False,
            use_layer_norm=True,
        )

        self.assertIsInstance(block.ln, torch.nn.LayerNorm)
        self.assertEqual(block.ln.normalized_shape, (self.out_features,))
        self.assertFalse(hasattr(block, "bn"))

    def test_initialization_with_gelu(self):
        """Test MLPBlock initialization with GELU activation."""
        block = MLPBlock(self.in_features, self.out_features, activation="gelu")
        self.assertIsInstance(block.act, torch.nn.GELU)

    def test_forward_shape(self):
        """Test MLPBlock forward pass output shape."""
        block = MLPBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features)
        y = block(x)

        self.assertEqual(y.shape, (self.batch_size, self.out_features))

    def test_forward_values(self):
        """Test MLPBlock forward pass produces non-zero values."""
        block = MLPBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features)
        y = block(x)

        self.assertGreater(y.abs().sum(), 0)

    def test_gradient_flow(self):
        """Test gradient flow through MLPBlock."""
        block = MLPBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in block.parameters():
            self.assertIsNotNone(param.grad)


class TestMLP(unittest.TestCase):
    """Unit tests for MLP class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 121
        self.output_channel = 120
        self.seq_length = 10
        self.hidden_sizes = [256, 128]

    def test_initialization_default(self):
        """Test MLP initialization with default parameters."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
        )

        self.assertEqual(model.feature_channel, self.feature_channel)
        self.assertEqual(model.output_channel, self.output_channel)
        self.assertEqual(model.seq_length, self.seq_length)
        self.assertEqual(model.hidden_sizes, [512, 256, 128])
        self.assertIsNotNone(model.positional_embed)
        self.assertEqual(len(model.layers), 3)

    def test_initialization_custom(self):
        """Test MLP initialization with custom parameters."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
            dropout=0.2,
            use_batch_norm=False,
            use_layer_norm=True,
            use_residual=True,
            activation="gelu",
            use_positional_embedding=False,
        )

        self.assertEqual(model.hidden_sizes, self.hidden_sizes)
        self.assertEqual(len(model.layers), 2)
        self.assertIsNone(model.positional_embed)
        self.assertTrue(model.use_residual)

    def test_forward_shape(self):
        """Test MLP forward pass output shape."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

    def test_forward_with_positional_embedding(self):
        """Test MLP forward with positional embedding enabled."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
            use_positional_embedding=True,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_without_positional_embedding(self):
        """Test MLP forward without positional embedding."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
            use_positional_embedding=False,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_residual(self):
        """Test MLP forward with residual connections."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
            use_residual=True,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_different_batch_sizes(self):
        """Test MLP forward with different batch sizes."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        batch_sizes = [4, 16, 32]

        for bs in batch_sizes:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = model(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

    def test_forward_different_sequence_lengths(self):
        """Test MLP forward with different sequence lengths."""
        seq_lengths = [5, 10, 20]

        for sl in seq_lengths:
            model = MLP(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                seq_length=sl,
                hidden_sizes=self.hidden_sizes,
            )

            x = torch.randn(self.batch_size, self.feature_channel, sl)
            y = model(x)
            self.assertEqual(y.shape, (self.batch_size, self.output_channel, sl))

    def test_gradient_flow(self):
        """Test gradient flow through MLP."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        x = torch.randn(
            self.batch_size, self.feature_channel, self.seq_length, requires_grad=True
        )
        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        x1 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y1 = model(x1)
        loss1 = y1.sum()
        loss1.backward()

        grad1 = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        x2 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y2 = model(x2)
        loss2 = y2.sum()
        loss2.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertIsNotNone(grad1.get(name))
                self.assertNotEqual(param.grad.sum(), grad1[name].sum())

    def test_model_parameters_count(self):
        """Test model has parameters."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        import tempfile
        import os

        model1 = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )
        model2.load_state_dict(torch.load(temp_file))

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)

        torch.testing.assert_close(y1, y2)

        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test model can be moved to CUDA if available."""
        model = MLP(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_sizes=self.hidden_sizes,
        )

        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


class TestMLPResidual(unittest.TestCase):
    """Unit tests for MLPResidual class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 121
        self.output_channel = 120
        self.seq_length = 10
        self.hidden_size = 128
        self.num_layers = 3

    def test_initialization(self):
        """Test MLPResidual initialization."""
        model = MLPResidual(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.assertEqual(len(model.blocks), self.num_layers)
        self.assertIsInstance(model.input_proj, torch.nn.Linear)
        self.assertIsInstance(model.output_proj, torch.nn.Linear)

    def test_forward_shape(self):
        """Test MLPResidual forward pass output shape."""
        model = MLPResidual(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

    def test_gradient_flow(self):
        """Test gradient flow through MLPResidual."""
        model = MLPResidual(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        x = torch.randn(
            self.batch_size, self.feature_channel, self.seq_length, requires_grad=True
        )
        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestMLPIntegration(unittest.TestCase):
    """Integration tests for MLP models."""

    def test_mlp_with_verticalrt_comparison(self):
        """Test MLP with a simple comparison to PINN."""
        from rtnn.models.pinn import PINN

        batch_size = 16
        feature_channel = 121
        output_channel = 120
        seq_length = 10

        mlp = MLP(
            feature_channel=feature_channel,
            output_channel=output_channel,
            seq_length=seq_length,
            hidden_sizes=[256, 128],
        )

        vrtn = PINN(
            feature_channel=feature_channel,
            hidden=64,
            out_channel=output_channel,
            n_layers=seq_length,
        )

        x = torch.randn(batch_size, feature_channel, seq_length)

        with torch.no_grad():
            y_mlp = mlp(x)
            y_vrtn = vrtn(x)

        self.assertEqual(y_mlp.shape, y_vrtn.shape)

    def test_mlp_batch_processing(self):
        """Test MLP with large batch processing."""
        model = MLP(
            feature_channel=121,
            output_channel=120,
            seq_length=10,
            hidden_sizes=[256, 128],
        )
        model.eval()

        large_batch = 16 * 32
        x = torch.randn(large_batch, 121, 10)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (large_batch, 120, 10))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMLPBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestMLP))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPResidual))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
