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
    """Unit tests for unnorm_mpas function."""

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
        self.idxmap = {0: "test_var"}

    def test_standard_normalization(self):
        """Test standard normalization de-normalization."""
        norm_type = {"test_var": "standard"}
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])  # (1,1,3)
        targ = torch.tensor([[[0.0, 0.5, 1.0]]])

        upred, utarg = unnorm_mpas(
            pred, targ, self.norm_mapping, norm_type, self.idxmap
        )

        # (x * std + mean): 0*2+5=5, 0.5*2+5=6, 1*2+5=7
        expected = torch.tensor([[[5.0, 6.0, 7.0]]])
        torch.testing.assert_close(upred, expected)
        torch.testing.assert_close(utarg, expected)

    def test_minmax_normalization(self):
        """Test minmax normalization de-normalization."""
        norm_type = {"test_var": "minmax"}
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])
        targ = torch.tensor([[[0.0, 0.5, 1.0]]])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # x * (max-min) + min: 0*10+0=0, 0.5*10+0=5, 1*10+0=10
        expected = torch.tensor([[[0.0, 5.0, 10.0]]])
        torch.testing.assert_close(upred, expected)

    def test_robust_normalization(self):
        """Test robust normalization de-normalization."""
        norm_type = {"test_var": "robust"}
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])
        targ = torch.tensor([[[0.0, 0.5, 1.0]]])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # x * iqr + median: 0*4+5=5, 0.5*4+5=7, 1*4+5=9
        expected = torch.tensor([[[5.0, 7.0, 9.0]]])
        torch.testing.assert_close(upred, expected)

    def test_log1p_standard_normalization(self):
        """Test log1p standard normalization de-normalization."""
        norm_type = {"test_var": "log1p_standard"}
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])
        targ = torch.tensor([[[0.0, 0.5, 1.0]]])

        upred, _ = unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)

        # unnorm = expm1(x * std + mean)
        # 0: expm1(0*0.3+0.5)=expm1(0.5)=0.6487
        # 0.5: expm1(0.5*0.3+0.5)=expm1(0.65)=0.9155
        # 1: expm1(1*0.3+0.5)=expm1(0.8)=1.2255
        self.assertEqual(upred.shape, (1, 1, 3))
        self.assertTrue(torch.all(upred >= 0))

    def test_unsupported_normalization(self):
        """Test unsupported normalization type raises error."""
        norm_type = {"test_var": "invalid_type"}
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])
        targ = torch.tensor([[[0.0, 0.5, 1.0]]])

        with self.assertRaises(ValueError):
            unnorm_mpas(pred, targ, self.norm_mapping, norm_type, self.idxmap)


class TestCalcHr(unittest.TestCase):
    """Unit tests for calc_hr function."""

    def setUp(self):
        """Set up test fixtures."""
        self.up = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        self.down = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]])

    def test_calc_hr_no_pressure(self):
        """Test calc_hr without pressure levels."""
        hr = calc_hr(self.up, self.down, p=None)

        # net = up - down = [0.5, 1.0, 1.5, 2.0]
        # dnet = net - roll(net) = [-1.5, 0.5, 0.5, 0.5]
        # hr = -dnet[:,:,1:] = -[0.5, 0.5, 0.5] = [-0.5, -0.5, -0.5]

        expected = torch.tensor([[[-0.5, -0.5, -0.5]]])

        self.assertEqual(hr.shape, (1, 1, 3))
        torch.testing.assert_close(hr, expected, rtol=1e-6, atol=1e-6)

    def test_calc_hr_with_pressure(self):
        """Test calc_hr with pressure levels."""
        p = torch.tensor([[[1000.0, 900.0, 800.0, 700.0]]])
        hr = calc_hr(self.up, self.down, p=p)

        self.assertEqual(hr.shape, (1, 1, 3))
        self.assertTrue(torch.all(hr < 0))

        # Check approximate values
        expected_approx = torch.tensor([[[-0.0422, -0.0422, -0.0422]]])
        torch.testing.assert_close(hr, expected_approx, rtol=1e-2, atol=1e-3)

    def test_calc_hr_zero_net(self):
        """Test calc_hr when up equals down."""
        up = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        down = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        hr = calc_hr(up, down)

        expected = torch.tensor([[[0.0, 0.0, 0.0]]])

        self.assertTrue(torch.all(hr == 0))
        torch.testing.assert_close(hr, expected)

    def test_calc_hr_with_constant_net(self):
        """Test calc_hr when net flux is constant."""
        up = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]])
        down = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        hr = calc_hr(up, down)

        expected = torch.tensor([[[0.0, 0.0, 0.0]]])
        torch.testing.assert_close(hr, expected, rtol=1e-6, atol=1e-6)

    def test_calc_hr_with_pressure_variable_dp(self):
        """Test calc_hr with non-uniform pressure spacing."""
        up = torch.tensor([[[1.0, 2.0, 4.0, 7.0]]])
        down = torch.tensor([[[0.5, 1.0, 2.0, 3.5]]])
        p = torch.tensor([[[1000.0, 850.0, 700.0, 500.0]]])

        hr = calc_hr(up, down, p=p)

        # net = [0.5, 1.0, 2.0, 3.5]
        # dnet = [-1.5, 0.5, 1.0, 1.5]
        # dp = [500, -150, -150, -200]
        # For indices 1:3, dnet/dp: 0.5/-150 = -0.00333, 1.0/-150 = -0.00667, 1.5/-200 = -0.0075
        # hr = dnet/dp * fac where fac ≈ 8.437
        # hr should be negative (since dnet positive, dp negative)

        self.assertEqual(hr.shape, (1, 1, 3))
        # All values should be negative (not positive as previously thought)
        self.assertTrue(torch.all(hr < 0))

    def test_calc_hr_2d_batch(self):
        """Test calc_hr with 2D batch (batch_size=2, channels=2)."""
        up = torch.tensor(
            [[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], [[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]]]
        )
        down = torch.tensor(
            [[[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]]
        )

        hr = calc_hr(up, down)

        # Expected shape: (batch=2, channels=2, seq_length-1=2)
        self.assertEqual(hr.shape, (2, 2, 2))
        self.assertIsInstance(hr, torch.Tensor)

        # Check first batch, first channel: net=[0.5,1.0,1.5], hr=-[0.5,0.5]=[-0.5,-0.5]
        expected_first = torch.tensor([-0.5, -0.5])
        torch.testing.assert_close(hr[0, 0], expected_first, rtol=1e-6, atol=1e-6)

        # Check first batch, second channel: net=[0.5,1.0,1.5], hr=[-0.5,-0.5]
        expected_second = torch.tensor([-0.5, -0.5])
        torch.testing.assert_close(hr[0, 1], expected_second, rtol=1e-6, atol=1e-6)

    def test_calc_hr_with_pressure_2d_batch(self):
        """Test calc_hr with pressure and 2D batch."""
        up = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0, 5.0]]])
        down = torch.tensor([[[0.5, 1.0, 1.5, 2.0]], [[1.0, 2.0, 3.0, 4.0]]])
        # p must match batch dimension
        p = torch.tensor(
            [[[1000.0, 900.0, 800.0, 700.0]], [[1000.0, 900.0, 800.0, 700.0]]]
        )

        hr = calc_hr(up, down, p=p)

        self.assertEqual(hr.shape, (2, 1, 3))
        self.assertIsInstance(hr, torch.Tensor)

        # Calculate expected for first batch: net=[0.5,1.0,1.5,2.0]
        # dnet = [-1.5, 0.5, 0.5, 0.5], dp = [300, -100, -100, -100]
        # hr = dnet/dp * fac ≈ 0.5/-100 * 8.437 = -0.0422
        expected_value = -0.0422
        # hr[0] has shape (1, 3), so expected should also have shape (1, 3)
        expected = torch.tensor([[expected_value, expected_value, expected_value]])

        # Check first batch (shape: 1, 3)
        torch.testing.assert_close(hr[0], expected, rtol=1e-2, atol=1e-3)

        # Second batch: net=[1.0,1.0,1.0,1.0] (since 2-1=1, 3-2=1, 4-3=1, 5-4=1)
        # dnet = [0, 0, 0, 0], so hr should be all zeros
        expected_zero = torch.tensor([[0.0, 0.0, 0.0]])
        torch.testing.assert_close(hr[1], expected_zero, rtol=1e-6, atol=1e-6)


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
