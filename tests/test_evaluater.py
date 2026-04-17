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
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtnn.evaluater import (
    NMSELoss,
    NMAELoss,
    MetricTracker,
    get_loss_function,
    mse_all,
    mbe_all,
    mae_all,
    r2_all,
    nmae_all,
    nmse_all,
    mare_all,
    gmrae_all,
    unnorm_mpas,
    calc_hr,
)


class TestNMSELoss(unittest.TestCase):
    """Unit tests for NMSELoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.criterion = NMSELoss(eps=1e-8)

    def test_forward(self):
        """Test NMSELoss forward pass."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_zero_loss(self):
        """Test NMSELoss with non-zero loss."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = self.criterion(pred, target)
        self.assertGreater(loss.item(), 0.0)

    def test_normalization(self):
        """Test that loss is normalized by target variance."""
        pred = torch.tensor([2.0, 4.0, 6.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        mse = torch.mean((pred - target) ** 2)
        norm = torch.mean(target**2)
        expected = mse / norm
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)


class TestNMAELoss(unittest.TestCase):
    """Unit tests for NMAELoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.criterion = NMAELoss(eps=1e-8)

    def test_forward(self):
        """Test NMAELoss forward pass."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_zero_loss(self):
        """Test NMAELoss with non-zero loss."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = self.criterion(pred, target)
        self.assertGreater(loss.item(), 0.0)


class TestMetricTracker(unittest.TestCase):
    """Unit tests for MetricTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MetricTracker()

    def test_metric_tracker_init(self):
        """Test MetricTracker initialization."""

        self.assertEqual(self.tracker.value, 0.0)
        self.assertEqual(self.tracker.count, 0)

    def test_metric_tracker_reset(self):
        """Test MetricTracker reset method."""

        tracker = MetricTracker()
        tracker.value = 10.5
        tracker.count = 5
        tracker.reset()
        self.assertEqual(tracker.value, 0.0)
        self.assertEqual(tracker.count, 0)

    def test_metric_tracker_update(self):
        """Test MetricTracker update method."""

        tracker = MetricTracker()

        # First update
        tracker.update(10.0, 5)
        self.assertEqual(tracker.value, 50.0)  # 10 * 5
        self.assertEqual(tracker.count, 5)

        # Second update
        tracker.update(20.0, 3)
        self.assertEqual(tracker.value, 110.0)  # 50 + 20*3
        self.assertEqual(tracker.count, 8)  # 5 + 3

        # Third update with zero count
        tracker.update(30.0, 0)
        self.assertEqual(tracker.value, 110.0)  # Unchanged
        self.assertEqual(tracker.count, 8)  # Unchanged

    def test_metric_tracker_getmean(self):
        """Test MetricTracker getmean method."""

        tracker = MetricTracker()

        # Test with valid updates
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        mean = tracker.getmean()
        expected_mean = 110.0 / 8  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
        self.assertAlmostEqual(mean, expected_mean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getmean()

    def test_metric_tracker_getstd(self):
        """Test MetricTracker getstd method."""

        tracker = MetricTracker()

        # Known values
        # Values: [10 (×5), 20 (×3)]
        # mean = 13.75
        # E[x^2] = (10^2 * 5 + 20^2 * 3) / 8 = (500 + 1200) / 8 = 212.5
        # variance = 212.5 - 13.75^2 = 23.4375
        # std = sqrt(23.4375) ≈ 4.841229
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        std = tracker.getstd()
        expected_std = np.sqrt(212.5 - 13.75**2)

        self.assertAlmostEqual(std, expected_std, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getstd()

    def test_metric_tracker_getsqrtmean(self):
        """Test MetricTracker getsqrtmean method."""

        tracker = MetricTracker()

        tracker.update(16.0, 2)  # mean = 16, sqrt = 4
        tracker.update(4.0, 2)  # mean = (16*2 + 4*2)/4 = 10, sqrt = sqrt(10)

        sqrtmean = tracker.getsqrtmean()
        expected_sqrtmean = np.sqrt(10.0)  # sqrt(10) ≈ 3.16227766
        self.assertAlmostEqual(sqrtmean, expected_sqrtmean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getsqrtmean()


class TestGetLossFunction(unittest.TestCase):
    """Unit tests for get_loss_function factory."""

    def test_mse_loss(self):
        """Test MSE loss creation."""
        args = Mock()
        loss = get_loss_function("mse", args)
        self.assertIsInstance(loss, torch.nn.MSELoss)

    def test_mae_loss(self):
        """Test MAE loss creation."""
        args = Mock()
        loss = get_loss_function("mae", args)
        self.assertIsInstance(loss, torch.nn.L1Loss)

    def test_nmae_loss(self):
        """Test NMAE loss creation."""
        args = Mock()
        loss = get_loss_function("nmae", args)
        self.assertIsInstance(loss, NMAELoss)

    def test_nmse_loss(self):
        """Test NMSE loss creation."""
        args = Mock()
        loss = get_loss_function("nmse", args)
        self.assertIsInstance(loss, NMSELoss)

    def test_huber_loss(self):
        """Test Huber loss creation."""
        args = Mock()
        args.beta_delta = 1.0
        loss = get_loss_function("huber", args)
        self.assertIsInstance(loss, torch.nn.HuberLoss)

    def test_smoothl1_loss(self):
        """Test SmoothL1 loss creation."""
        args = Mock()
        args.beta_delta = 1.0
        loss = get_loss_function("smoothl1", args)
        self.assertIsInstance(loss, torch.nn.SmoothL1Loss)

    def test_invalid_loss_type(self):
        """Test invalid loss type raises error."""
        args = Mock()
        with self.assertRaises(ValueError):
            get_loss_function("invalid_loss", args)

    def test_huber_without_beta_delta(self):
        """Test Huber loss without beta_delta raises error."""
        args = Mock()
        delattr(args, "beta_delta")
        with self.assertRaises(ValueError):
            get_loss_function("huber", args)


class TestMetricFunctions(unittest.TestCase):
    """Unit tests for metric functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.pred = torch.tensor([2.0, 3.0, 4.0])
        self.true = torch.tensor([1.0, 2.0, 3.0])

    def test_mse_all(self):
        """Test MSE metric."""
        count, value = mse_all(self.pred, self.true)
        expected_mse = (1**2 + 1**2 + 1**2) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mse, places=6)

    def test_mbe_all(self):
        """Test MBE metric."""
        count, value = mbe_all(self.pred, self.true)
        expected_mbe = (1 + 1 + 1) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mbe, places=6)

    def test_mae_all(self):
        """Test MAE metric."""
        count, value = mae_all(self.pred, self.true)
        expected_mae = (1 + 1 + 1) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mae, places=6)

    def test_r2_all(self):
        """Test R² metric."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        true = torch.tensor([2.0, 3.0, 4.0])
        count, value = r2_all(pred, true)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), 1.0, places=6)

    def test_r2_all_perfect(self):
        """Test R² metric with perfect predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 2.0, 3.0])
        count, value = r2_all(pred, true)
        self.assertAlmostEqual(value.item(), 1.0, places=6)

    def test_nmae_all(self):
        """Test NMAE metric."""
        count, value = nmae_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_nmse_all(self):
        """Test NMSE metric."""
        count, value = nmse_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_mare_all(self):
        """Test MARE metric."""
        count, value = mare_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_gmrae_all(self):
        """Test GMRAE metric."""
        count, value = gmrae_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)


class TestUnnormMpas(unittest.TestCase):
    """Unit tests for unnorm_mpas function with 5D tensors."""

    def setUp(self):
        """Set up test fixtures."""
        self.norm_mapping = {
            "test_var": {
                "vmin": 0,
                "vmax": 10,
                "vmean": 5,
                "vstd": 2,
                "median": 5,
                "iqr": 4,
                "log_min": 0,
                "log_max": 1,
                "log_mean": 0.5,
                "log_std": 0.3,
                "log_median": 0.5,
                "log_iqr": 0.4,
                "sqrt_min": 0,
                "sqrt_max": 3,
                "sqrt_mean": 1.5,
                "sqrt_std": 0.8,
                "sqrt_median": 1.5,
                "sqrt_iqr": 1.2,
            }
        }
        # idxmap maps variable index to variable name (4 variables total)
        self.idxmap = {0: "test_var", 1: "test_var", 2: "test_var", 3: "test_var"}

        # Test dimensions
        self.batch_size = 2
        self.n_pft = 3
        self.n_bands = 2
        self.seq_len = 5

    def test_standard_normalization(self):
        """Test standard normalization de-normalization with 5D tensors."""
        norm_type = {"test_var": "standard"}

        # Create 5D tensor: (batch, channels, n_pft, n_bands, seq)
        pred = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    pred[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                    targ[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        upred, utarg = unnorm_mpas(
            pred, targ, self.norm_mapping, norm_type, self.idxmap
        )

        # Expected: (x * std + mean): 0*2+5=5, 0.5*2+5=6, 1*2+5=7
        expected = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    expected[b, 0, pft, band, :] = torch.tensor([5.0, 6.0, 7.0])

        torch.testing.assert_close(upred, expected)
        torch.testing.assert_close(utarg, expected)

    def test_minmax_normalization(self):
        """Test minmax normalization de-normalization with 5D tensors."""
        norm_type = {"test_var": "minmax"}

        pred = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    pred[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                    targ[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # x * (max-min) + min: 0*10+0=0, 0.5*10+0=5, 1*10+0=10
        expected = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    expected[b, 0, pft, band, :] = torch.tensor([0.0, 5.0, 10.0])

        torch.testing.assert_close(upred, expected)

    def test_robust_normalization(self):
        """Test robust normalization de-normalization with 5D tensors."""
        norm_type = {"test_var": "robust"}

        pred = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    pred[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                    targ[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # x * iqr + median: 0*4+5=5, 0.5*4+5=7, 1*4+5=9
        expected = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    expected[b, 0, pft, band, :] = torch.tensor([5.0, 7.0, 9.0])

        torch.testing.assert_close(upred, expected)

    def test_log1p_standard_normalization(self):
        """Test log1p standard normalization de-normalization with 5D tensors."""
        norm_type = {"test_var": "log1p_standard"}

        pred = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    pred[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                    targ[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # Check shape
        self.assertEqual(upred.shape, (self.batch_size, 1, self.n_pft, self.n_bands, 3))
        self.assertTrue(torch.all(upred >= 0))

        # Check specific values for first batch, first pft, first band
        # unnorm = expm1(x * std + mean)
        # 0: expm1(0*0.3+0.5)=expm1(0.5)=0.6487
        # 0.5: expm1(0.5*0.3+0.5)=expm1(0.65)=0.9155
        # 1: expm1(1*0.3+0.5)=expm1(0.8)=1.2255
        expected_first = torch.tensor([0.6487, 0.9155, 1.2255])
        actual_first = upred[0, 0, 0, 0, :]
        torch.testing.assert_close(actual_first, expected_first, rtol=1e-4, atol=1e-4)

    def test_multiple_variables(self):
        """Test unnormalization with multiple variables (4 channels)."""
        norm_type = {"test_var": "standard"}

        # Create 5D tensor with 4 channels
        pred = torch.zeros(self.batch_size, 4, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 4, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for c in range(4):
                for pft in range(self.n_pft):
                    for band in range(self.n_bands):
                        pred[b, c, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                        targ[b, c, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # Each of the 4 channels should be correctly unnormalized to [5,6,7]
        expected_val = torch.tensor([5.0, 6.0, 7.0])

        self.assertEqual(upred.shape, pred.shape)
        for b in range(self.batch_size):
            for c in range(4):
                for pft in range(self.n_pft):
                    for band in range(self.n_bands):
                        # Use torch.allclose instead of assert_close to avoid rtol/atol requirement
                        self.assertTrue(
                            torch.allclose(
                                upred[b, c, pft, band, :],
                                expected_val,
                                rtol=1e-6,
                                atol=1e-6,
                            ),
                            f"Mismatch at batch={b}, channel={c}, pft={pft}, band={band}",
                        )

    def test_unsupported_normalization(self):
        """Test unsupported normalization type raises error."""
        norm_type = {"test_var": "invalid_type"}

        pred = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)
        targ = torch.zeros(self.batch_size, 1, self.n_pft, self.n_bands, 3)

        for b in range(self.batch_size):
            for pft in range(self.n_pft):
                for band in range(self.n_bands):
                    pred[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])
                    targ[b, 0, pft, band, :] = torch.tensor([0.0, 0.5, 1.0])

        with self.assertRaises(ValueError):
            unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)


class TestCalcHr(unittest.TestCase):
    """Unit tests for calc_hr function with 5D tensors."""

    def setUp(self):
        """Set up test fixtures with 5D tensors."""
        # Shape: (batch=1, channels=1, n_pft=1, n_bands=1, seq=4)
        self.up = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
        self.down = torch.tensor([[[[[0.5, 1.0, 1.5, 2.0]]]]])

        self.n_pft = 1
        self.n_bands = 1
        self.batch_size = 1
        self.n_chans = 1

    def test_calc_hr_no_pressure(self):
        """Test calc_hr without pressure levels."""
        hr = calc_hr(self.up, self.down, p=None)

        # net = up - down = [0.5, 1.0, 1.5, 2.0]
        # dnet = net - roll(net) = [-1.5, 0.5, 0.5, 0.5]
        # hr = -dnet[..., 1:] = -[0.5, 0.5, 0.5] = [-0.5, -0.5, -0.5]

        expected = torch.tensor([[[[[-0.5, -0.5, -0.5]]]]])

        self.assertEqual(hr.shape, (1, 1, 1, 1, 3))
        torch.testing.assert_close(hr, expected, rtol=1e-6, atol=1e-6)

    def test_calc_hr_zero_net(self):
        """Test calc_hr when up equals down."""
        up = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
        down = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
        hr = calc_hr(up, down)

        expected = torch.tensor([[[[[0.0, 0.0, 0.0]]]]])

        self.assertTrue(torch.all(hr == 0))
        torch.testing.assert_close(hr, expected)

    def test_calc_hr_with_constant_net(self):
        """Test calc_hr when net flux is constant."""
        up = torch.tensor([[[[[2.0, 2.0, 2.0, 2.0]]]]])
        down = torch.tensor([[[[[1.0, 1.0, 1.0, 1.0]]]]])
        hr = calc_hr(up, down)

        expected = torch.tensor([[[[[0.0, 0.0, 0.0]]]]])
        torch.testing.assert_close(hr, expected, rtol=1e-6, atol=1e-6)

    def test_calc_hr_with_pft_and_bands(self):
        """Test calc_hr with multiple PFTs and bands."""
        # Shape: (batch=1, channels=1, n_pft=2, n_bands=2, seq=4)
        up = torch.zeros(1, 1, 2, 2, 4)
        down = torch.zeros(1, 1, 2, 2, 4)

        for pft in range(2):
            for band in range(2):
                up[0, 0, pft, band, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
                down[0, 0, pft, band, :] = torch.tensor([0.5, 1.0, 1.5, 2.0])

        hr = calc_hr(up, down, p=None)

        # Each PFT and band should have the same result
        self.assertEqual(hr.shape, (1, 1, 2, 2, 3))
        expected_val = torch.tensor([-0.5, -0.5, -0.5])

        for pft in range(2):
            for band in range(2):
                torch.testing.assert_close(
                    hr[0, 0, pft, band, :], expected_val, rtol=1e-6, atol=1e-6
                )

    def test_calc_hr_2d_batch(self):
        """Test calc_hr with 2D batch (batch_size=2, channels=2, n_pft=1, n_bands=1)."""
        # Shape: (batch=2, channels=2, n_pft=1, n_bands=1, seq=3)
        up = torch.zeros(2, 2, 1, 1, 3)
        down = torch.zeros(2, 2, 1, 1, 3)

        # Batch 0, Channel 0
        up[0, 0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0])
        down[0, 0, 0, 0, :] = torch.tensor([0.5, 1.0, 1.5])
        # Batch 0, Channel 1
        up[0, 1, 0, 0, :] = torch.tensor([1.5, 2.5, 3.5])
        down[0, 1, 0, 0, :] = torch.tensor([1.0, 1.5, 2.0])
        # Batch 1, Channel 0
        up[1, 0, 0, 0, :] = torch.tensor([2.0, 3.0, 4.0])
        down[1, 0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0])
        # Batch 1, Channel 1
        up[1, 1, 0, 0, :] = torch.tensor([2.5, 3.5, 4.5])
        down[1, 1, 0, 0, :] = torch.tensor([1.5, 2.5, 3.5])

        hr = calc_hr(up, down, p=None)

        # Expected shape: (2, 2, 1, 1, 2)
        self.assertEqual(hr.shape, (2, 2, 1, 1, 2))
        self.assertIsInstance(hr, torch.Tensor)

        # Check first batch, first channel: net=[0.5,1.0,1.5], hr=-[0.5,0.5]=[-0.5,-0.5]
        expected_first = torch.tensor([-0.5, -0.5])
        torch.testing.assert_close(
            hr[0, 0, 0, 0, :], expected_first, rtol=1e-6, atol=1e-6
        )

        # Check first batch, second channel: net=[0.5,1.0,1.5], hr=[-0.5,-0.5]
        expected_second = torch.tensor([-0.5, -0.5])
        torch.testing.assert_close(
            hr[0, 1, 0, 0, :], expected_second, rtol=1e-6, atol=1e-6
        )

    def test_calc_hr_single_element(self):
        """Test calc_hr with single element sequence (should return empty)."""
        up = torch.tensor([[[[[1.0]]]]])
        down = torch.tensor([[[[[0.5]]]]])
        hr = calc_hr(up, down, p=None)

        # With seq_length=1, there are no valid derivatives
        self.assertEqual(hr.shape, (1, 1, 1, 1, 0))

    def test_calc_hr_multiple_pft_bands_different_values(self):
        """Test calc_hr with different values across PFTs and bands."""
        # Shape: (batch=1, channels=1, n_pft=2, n_bands=2, seq=4)
        up = torch.zeros(1, 1, 2, 2, 4)
        down = torch.zeros(1, 1, 2, 2, 4)

        # Different values for each combination
        up[0, 0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        down[0, 0, 0, 0, :] = torch.tensor([0.5, 1.0, 1.5, 2.0])

        up[0, 0, 0, 1, :] = torch.tensor([2.0, 4.0, 6.0, 8.0])
        down[0, 0, 0, 1, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        up[0, 0, 1, 0, :] = torch.tensor([1.5, 3.0, 4.5, 6.0])
        down[0, 0, 1, 0, :] = torch.tensor([0.75, 1.5, 2.25, 3.0])

        up[0, 0, 1, 1, :] = torch.tensor([3.0, 6.0, 9.0, 12.0])
        down[0, 0, 1, 1, :] = torch.tensor([1.5, 3.0, 4.5, 6.0])

        hr = calc_hr(up, down, p=None)

        self.assertEqual(hr.shape, (1, 1, 2, 2, 3))

        # Each combination should have hr = [-0.5, -0.5, -0.5] * (net_scale)
        # net = up - down, then hr = -dnet[..., 1:]
        # For constant step size, hr should be constant across sequence
        for pft in range(2):
            for band in range(2):
                # Check that all values in the sequence are the same (constant derivative)
                hr_seq = hr[0, 0, pft, band, :]
                self.assertTrue(
                    torch.allclose(hr_seq[0], hr_seq[1]),
                    f"hr not constant for pft={pft}, band={band}",
                )
                self.assertTrue(
                    torch.allclose(hr_seq[0], hr_seq[2]),
                    f"hr not constant for pft={pft}, band={band}",
                )


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestNMSELoss))
    suite.addTests(loader.loadTestsFromTestCase(TestNMAELoss))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestGetLossFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestUnnormMpas))
    suite.addTests(loader.loadTestsFromTestCase(TestCalcHr))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
