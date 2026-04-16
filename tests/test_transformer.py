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

from rtnn.models.Transformer import SelfAttention, TransformerBlock, Encoder
from rtnn.models.Transformer import EncoderTorch
from rtnn.model_utils import ModelUtils


class TestSelfAttention(unittest.TestCase):
    """Unit tests for SelfAttention class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.seq_len = 10
        self.embed_size = 128
        self.heads = 4
        self.attention = SelfAttention(self.embed_size, self.heads)

    def test_initialization(self):
        """Test SelfAttention initialization."""
        self.assertEqual(self.attention.embed_size, self.embed_size)
        self.assertEqual(self.attention.heads, self.heads)
        self.assertEqual(self.attention.head_dim, self.embed_size // self.heads)
        self.assertIsInstance(self.attention.values, torch.nn.Linear)
        self.assertIsInstance(self.attention.keys, torch.nn.Linear)
        self.assertIsInstance(self.attention.queries, torch.nn.Linear)
        self.assertIsInstance(self.attention.fc_out, torch.nn.Linear)

    def test_forward_shape(self):
        """Test SelfAttention forward output shape."""
        values = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        keys = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        query = torch.randn(self.batch_size, self.seq_len, self.embed_size)

        out = self.attention(values, keys, query, mask=None)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_size))

    def test_forward_with_mask(self):
        """Test SelfAttention forward with mask."""
        values = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        keys = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        query = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len)

        out = self.attention(values, keys, query, mask)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_size))


class TestTransformerBlock(unittest.TestCase):
    """Unit tests for TransformerBlock class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.seq_len = 10
        self.embed_size = 128
        self.heads = 4
        self.dropout = 0.1
        self.forward_expansion = 4
        self.block = TransformerBlock(
            self.embed_size, self.heads, self.dropout, self.forward_expansion
        )

    def test_initialization(self):
        """Test TransformerBlock initialization."""
        self.assertIsInstance(self.block.attention, SelfAttention)
        self.assertIsInstance(self.block.norm1, torch.nn.LayerNorm)
        self.assertIsInstance(self.block.norm2, torch.nn.LayerNorm)
        self.assertIsInstance(self.block.feed_forward, torch.nn.Sequential)
        self.assertIsInstance(self.block.dropout, torch.nn.Dropout)

    def test_forward_shape(self):
        """Test TransformerBlock forward output shape."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_size)
        out = self.block(x, x, x, mask=None)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.embed_size))


class TestEncoder(unittest.TestCase):
    """Unit tests for Encoder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 6
        self.output_channel = 4
        self.embed_size = 64
        self.num_layers = 2
        self.heads = 4
        self.forward_expansion = 4
        self.seq_length = 10
        self.dropout = 0.1

        self.encoder = Encoder(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
        )

    def test_initialization(self):
        """Test Encoder initialization."""
        self.assertEqual(self.encoder.embed_size, self.embed_size)
        self.assertEqual(self.encoder.seq_length, self.seq_length)
        self.assertIsInstance(self.encoder.first, torch.nn.Linear)
        self.assertIsInstance(self.encoder.first_act, torch.nn.ReLU)
        self.assertIsInstance(self.encoder.position_embedding, torch.nn.Embedding)
        self.assertEqual(len(self.encoder.layers), self.num_layers)
        self.assertIsInstance(self.encoder.dropout, torch.nn.Dropout)
        self.assertIsInstance(self.encoder.final, torch.nn.Conv1d)

    def test_forward_shape(self):
        """Test Encoder forward output shape."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = self.encoder(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_mask(self):
        """Test Encoder forward with mask."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_length)
        y = self.encoder(x, mask)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_different_batch_sizes(self):
        """Test Encoder forward with different batch sizes."""
        batch_sizes = [1, 4, 16, 32]

        for bs in batch_sizes:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = self.encoder(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        import tempfile
        import os

        model1 = self.encoder

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        # Load model
        model2 = Encoder(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
        )
        model2.load_state_dict(torch.load(temp_file))

        # Test forward pass produces same output
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)

        torch.testing.assert_close(y1, y2)

        # Cleanup
        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test model can be moved to CUDA if available."""
        if torch.cuda.is_available():
            model = self.encoder.cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


class TestEncoderIntegration(unittest.TestCase):
    """Integration tests for Encoder."""

    def test_encoder_with_different_sequence_lengths(self):
        """Test encoder with different sequence lengths."""
        seq_lengths = [5, 10, 20]

        for sl in seq_lengths:
            encoder = Encoder(
                feature_channel=6,
                output_channel=4,
                embed_size=64,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=sl,
                dropout=0.1,
            )
            x = torch.randn(16, 6, sl)
            y = encoder(x)
            self.assertEqual(y.shape, (16, 4, sl))

    def test_encoder_with_different_embed_sizes(self):
        """Test encoder with different embedding sizes."""
        embed_sizes = [32, 64, 128]

        for es in embed_sizes:
            encoder = Encoder(
                feature_channel=6,
                output_channel=4,
                embed_size=es,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=10,
                dropout=0.1,
            )
            x = torch.randn(16, 6, 10)
            y = encoder(x)
            self.assertEqual(y.shape, (16, 4, 10))


class TestEncoderTorch(unittest.TestCase):
    """Unit tests for EncoderTorch (PyTorch-based implementation)."""

    def setUp(self):
        self.batch_size = 32
        self.feature_channel = 6
        self.output_channel = 4
        self.embed_size = 128
        self.num_layers = 3
        self.heads = 8
        self.forward_expansion = 4
        self.seq_length = 10
        self.dropout = 0.1

        self.encoder = EncoderTorch(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
        )
        self.param_count = ModelUtils.get_parameter_number(self.encoder, None)
        print(
            f"Model Parameters - Total: {self.param_count['Total']:,}, Trainable: {self.param_count['Trainable']:,}"
        )

    def test_initialization(self):
        """Test EncoderTorch initialization."""
        self.assertEqual(self.encoder.embed_size, self.embed_size)
        self.assertEqual(self.encoder.seq_length, self.seq_length)

        self.assertIsInstance(self.encoder.input_proj, torch.nn.Linear)
        self.assertIsInstance(self.encoder.position_embedding, torch.nn.Embedding)
        self.assertIsInstance(self.encoder.encoder, torch.nn.TransformerEncoder)
        self.assertIsInstance(self.encoder.dropout, torch.nn.Dropout)
        self.assertIsInstance(self.encoder.final, torch.nn.Conv1d)

    def test_forward_shape(self):
        """Test forward output shape."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = self.encoder(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_padding_mask(self):
        """Test forward with src_key_padding_mask."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)

        # mask shape: (batch, seq_len)
        padding_mask = torch.zeros(self.batch_size, self.seq_length).bool()

        y = self.encoder(x, src_key_padding_mask=padding_mask)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_attention_mask(self):
        """Test forward with attention mask (causal or full)."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)

        # mask shape: (seq_len, seq_len)
        attn_mask = torch.zeros(self.seq_length, self.seq_length)

        y = self.encoder(x, mask=attn_mask)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_different_batch_sizes(self):
        """Test different batch sizes."""
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = self.encoder(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

    def test_model_serialization(self):
        """Test save/load consistency."""
        import tempfile

        model1 = self.encoder

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = EncoderTorch(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
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
        """Test CUDA transfer."""
        if torch.cuda.is_available():
            model = self.encoder.cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()

            y = model(x)
            self.assertTrue(y.is_cuda)


class TestEncoderTorchIntegration(unittest.TestCase):
    """Integration tests."""

    def test_different_sequence_lengths(self):
        """Test variable sequence lengths."""
        for sl in [5, 10, 20]:
            encoder = EncoderTorch(
                feature_channel=6,
                output_channel=4,
                embed_size=64,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=sl,
                dropout=0.1,
            )

            x = torch.randn(16, 6, sl)
            y = encoder(x)

            self.assertEqual(y.shape, (16, 4, sl))

    def test_different_embed_sizes(self):
        """Test variable embedding sizes."""
        for es in [32, 64, 128]:
            encoder = EncoderTorch(
                feature_channel=6,
                output_channel=4,
                embed_size=es,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=10,
                dropout=0.1,
            )

            x = torch.randn(16, 6, 10)
            y = encoder(x)

            self.assertEqual(y.shape, (16, 4, 10))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSelfAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderTorch))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderTorchIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
