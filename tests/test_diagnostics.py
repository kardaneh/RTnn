# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import numpy as np
import os
import shutil
from unittest.mock import Mock
import sys

# Import the module to test
from rtnn.diagnostics import (
    stats,
    plot_flux_and_abs_lines,
    plot_flux_and_abs,
    plot_metric_histories,
    plot_loss_histories,
    plot_spatial_temporal_density,
    plot_all_diagnostics,
)


class TestDiagnostics(unittest.TestCase):
    """Unit tests for diagnostics plotting functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create tests_plots directory
        self.plots_dir = os.path.join(os.path.dirname(__file__), "tests_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Create temporary directory for stats test
        self.temp_dir = os.path.join(self.plots_dir, "temp_stats")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create dummy data - use larger batch for tests that need 10+ samples
        self.batch_size = 32  # Increased from 8 to 32 for line plots
        self.seq_len = 10
        self.n_pft = 15
        self.n_bands = 2
        self.n_chans = 4
        self.n_outputs = self.n_chans * self.n_pft * self.n_bands  # 120

        # Create dummy predictions and targets
        self.predicts = torch.randn(self.batch_size, self.n_outputs, self.seq_len)
        self.targets = torch.randn(self.batch_size, self.n_outputs, self.seq_len)

        # Create dummy absorption data
        self.abs12_predict = torch.randn(self.batch_size, 1, self.seq_len - 1)
        self.abs12_target = torch.randn(self.batch_size, 1, self.seq_len - 1)
        self.abs34_predict = torch.randn(self.batch_size, 1, self.seq_len - 1)
        self.abs34_target = torch.randn(self.batch_size, 1, self.seq_len - 1)

        # Create dummy trackers
        self.sindex_tracker = np.random.randint(0, 16, 1000)
        self.tindex_tracker = np.random.randint(0, 100, 1000)

        # Create dummy metric histories
        self.train_history = {
            "fluxes_NMAE": [0.5, 0.4, 0.3],
            "fluxes_NMSE": [0.3, 0.25, 0.2],
            "fluxes_R2": [0.7, 0.8, 0.85],
            "abs12_NMAE": [0.6, 0.5, 0.4],
            "abs12_NMSE": [0.4, 0.35, 0.3],
            "abs12_R2": [0.6, 0.7, 0.75],
        }
        self.valid_history = {
            "fluxes_NMAE": [0.55, 0.45, 0.35],
            "fluxes_NMSE": [0.35, 0.3, 0.25],
            "fluxes_R2": [0.65, 0.75, 0.8],
            "abs12_NMAE": [0.65, 0.55, 0.45],
            "abs12_NMSE": [0.45, 0.4, 0.35],
            "abs12_R2": [0.55, 0.65, 0.7],
        }

        # Create dummy logger
        self.logger = Mock()

        # Create dummy NetCDF file for stats test
        import xarray as xr
        import pandas as pd

        self.test_nc_file = os.path.join(self.temp_dir, "test.nc")
        time = pd.date_range("2000-01-01", periods=10, freq="D")
        ds = xr.Dataset(
            {
                "test_var1": (("time",), np.random.randn(10)),
                "test_var2": (("time",), np.random.randn(10) * 10),
            },
            coords={"time": time},
        )
        ds.to_netcdf(self.test_nc_file)
        ds.close()

    def tearDown(self):
        """Clean up temporary files (keep tests_plots directory)."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------------------
    # stats function tests
    # ------------------------------------------------------------------------

    def test_stats_computation(self):
        """Test stats function computes statistics correctly."""
        norm_mapping = stats(
            file_list=[self.test_nc_file],
            logger=self.logger,
            output_dir=self.plots_dir,
            norm_mapping=None,
            plots=False,
        )

        self.assertIn("test_var1", norm_mapping)
        self.assertIn("test_var2", norm_mapping)

        expected_keys = [
            "vmin",
            "vmax",
            "vmean",
            "vstd",
            "q1",
            "q3",
            "iqr",
            "median",
            "log_min",
            "log_max",
            "log_mean",
            "log_std",
            "log_q1",
            "log_q3",
            "log_iqr",
            "log_median",
            "sqrt_min",
            "sqrt_max",
            "sqrt_mean",
            "sqrt_std",
            "sqrt_q1",
            "sqrt_q3",
            "sqrt_iqr",
            "sqrt_median",
        ]
        for key in expected_keys:
            self.assertIn(key, norm_mapping["test_var1"])

    def test_stats_with_plots(self):
        """Test stats function generates histogram plots."""
        _ = stats(
            file_list=[self.test_nc_file],
            logger=self.logger,
            output_dir=self.plots_dir,
            norm_mapping=None,
            plots=True,
        )

        self.assertTrue(
            os.path.exists(os.path.join(self.plots_dir, "test_var1_histogram.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.plots_dir, "test_var2_histogram.png"))
        )

    def test_stats_updates_existing_mapping(self):
        """Test stats function updates existing norm_mapping."""
        existing_mapping = {"existing_var": {"vmean": 1.0}}

        norm_mapping = stats(
            file_list=[self.test_nc_file],
            logger=self.logger,
            output_dir=self.temp_dir,
            norm_mapping=existing_mapping,
            plots=False,
        )

        self.assertIn("existing_var", norm_mapping)
        self.assertIn("test_var1", norm_mapping)

    # ------------------------------------------------------------------------
    # plot_flux_and_abs_lines tests
    # ------------------------------------------------------------------------

    def test_plot_flux_and_abs_lines_basic(self):
        """Test basic line plot generation."""
        output_file = os.path.join(self.plots_dir, "test_lines_basic.png")

        predicts_4ch = self.predicts[:, :4, :]
        targets_4ch = self.targets[:, :4, :]

        plot_flux_and_abs_lines(
            predicts_4ch,
            targets_4ch,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    def test_plot_flux_and_abs_lines_with_absorption(self):
        """Test line plot with absorption panels."""
        output_file = os.path.join(self.plots_dir, "test_lines_with_abs.png")

        predicts_4ch = self.predicts[:, :4, :]
        targets_4ch = self.targets[:, :4, :]

        plot_flux_and_abs_lines(
            predicts_4ch,
            targets_4ch,
            abs12_predict=self.abs12_predict,
            abs12_target=self.abs12_target,
            abs34_predict=self.abs34_predict,
            abs34_target=self.abs34_target,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_flux_and_abs tests
    # ------------------------------------------------------------------------

    def test_plot_flux_and_abs_basic(self):
        """Test basic hexbin plot generation."""
        output_file = os.path.join(self.plots_dir, "test_hexbin_basic.png")

        predicts_4ch = self.predicts[:, :4, :]
        targets_4ch = self.targets[:, :4, :]

        plot_flux_and_abs(
            predicts_4ch,
            targets_4ch,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    def test_plot_flux_and_abs_with_absorption(self):
        """Test hexbin plot with absorption panels."""
        output_file = os.path.join(self.plots_dir, "test_hexbin_with_abs.png")

        predicts_4ch = self.predicts[:, :4, :]
        targets_4ch = self.targets[:, :4, :]

        plot_flux_and_abs(
            predicts_4ch,
            targets_4ch,
            abs12_predict=self.abs12_predict,
            abs12_target=self.abs12_target,
            abs34_predict=self.abs34_predict,
            abs34_target=self.abs34_target,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_metric_histories tests
    # ------------------------------------------------------------------------

    def test_plot_metric_histories(self):
        """Test metric histories plot generation."""
        output_file = os.path.join(self.plots_dir, "test_metrics.png")

        plot_metric_histories(
            self.train_history,
            self.valid_history,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    def test_plot_metric_histories_empty(self):
        """Test metric histories with empty data."""
        output_file = os.path.join(self.plots_dir, "test_metrics_empty.png")

        # Should not raise error
        plot_metric_histories(
            {},
            {},
            filename=output_file,
            logger=self.logger,
        )

        # File should not be created for empty data
        self.assertFalse(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_loss_histories tests
    # ------------------------------------------------------------------------

    def test_plot_loss_histories(self):
        """Test loss histories plot generation."""
        output_file = os.path.join(self.plots_dir, "test_loss.png")
        train_loss = [0.5, 0.4, 0.3, 0.25]
        valid_loss = [0.55, 0.45, 0.35, 0.3]

        plot_loss_histories(
            train_loss,
            valid_loss,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_spatial_temporal_density tests
    # ------------------------------------------------------------------------

    def test_plot_spatial_temporal_density(self):
        """Test spatial-temporal density plot generation."""
        plot_spatial_temporal_density(
            self.sindex_tracker,
            self.tindex_tracker,
            mode="test",
            save_dir=self.plots_dir,
            filename="test_density",
            logger=self.logger,
        )

        output_file = os.path.join(self.plots_dir, "test_density_test.png")
        self.assertIsNotNone(output_file)
        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_all_diagnostics tests
    # ------------------------------------------------------------------------

    def test_plot_all_diagnostics(self):
        """Test all diagnostics plot generation."""
        batch = self.batch_size
        predicts_5d = self.predicts.reshape(
            batch, self.n_chans, self.n_pft, self.n_bands, self.seq_len
        )
        targets_5d = self.targets.reshape(
            batch, self.n_chans, self.n_pft, self.n_bands, self.seq_len
        )

        # Tile absorption data to match PFT and band dimensions
        abs12_predict_tiled = (
            self.abs12_predict.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, self.n_pft, self.n_bands, -1)
        )
        abs12_target_tiled = (
            self.abs12_target.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, self.n_pft, self.n_bands, -1)
        )
        abs34_predict_tiled = (
            self.abs34_predict.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, self.n_pft, self.n_bands, -1)
        )
        abs34_target_tiled = (
            self.abs34_target.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, self.n_pft, self.n_bands, -1)
        )

        diag_dir = os.path.join(self.plots_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        plot_all_diagnostics(
            predicts_5d,
            targets_5d,
            abs12_predict=abs12_predict_tiled,
            abs12_target=abs12_target_tiled,
            abs34_predict=abs34_predict_tiled,
            abs34_target=abs34_target_tiled,
            n_pft=self.n_pft,
            n_bands=self.n_bands,
            output_dir=diag_dir,
            prefix="test_diagnostics",
            logger=self.logger,
        )

        files = os.listdir(diag_dir)
        png_files = [f for f in files if f.endswith(".png")]
        self.assertGreater(len(png_files), 0)

    def test_plot_all_diagnostics_no_absorption(self):
        """Test all diagnostics without absorption data."""
        batch = self.batch_size
        predicts_5d = self.predicts.reshape(
            batch, self.n_chans, self.n_pft, self.n_bands, self.seq_len
        )
        targets_5d = self.targets.reshape(
            batch, self.n_chans, self.n_pft, self.n_bands, self.seq_len
        )

        diag_dir = os.path.join(self.plots_dir, "diagnostics_no_abs")
        os.makedirs(diag_dir, exist_ok=True)

        plot_all_diagnostics(
            predicts_5d,
            targets_5d,
            abs12_predict=None,
            abs12_target=None,
            abs34_predict=None,
            abs34_target=None,
            n_pft=self.n_pft,
            n_bands=self.n_bands,
            output_dir=diag_dir,
            prefix="test_diagnostics_no_abs",
            logger=self.logger,
        )

        files = os.listdir(diag_dir)
        png_files = [f for f in files if f.endswith(".png")]
        self.assertGreater(len(png_files), 0)

    # ------------------------------------------------------------------------
    # Integration tests
    # ------------------------------------------------------------------------

    def test_all_plots_generate_files(self):
        """Test that all plot functions generate files without errors."""
        output_files = []

        # Test line plot
        f1 = os.path.join(self.plots_dir, "integration_line.png")
        predicts_4ch = self.predicts[:, :4, :]
        targets_4ch = self.targets[:, :4, :]
        plot_flux_and_abs_lines(
            predicts_4ch, targets_4ch, filename=f1, logger=self.logger
        )
        output_files.append(f1)

        # Test hexbin plot
        f2 = os.path.join(self.plots_dir, "integration_hexbin.png")
        plot_flux_and_abs(predicts_4ch, targets_4ch, filename=f2, logger=self.logger)
        output_files.append(f2)

        # Test metric histories
        f3 = os.path.join(self.plots_dir, "integration_metrics.png")
        plot_metric_histories(
            self.train_history, self.valid_history, filename=f3, logger=self.logger
        )
        output_files.append(f3)

        # Test loss histories
        f4 = os.path.join(self.plots_dir, "integration_loss.png")
        plot_loss_histories([0.5, 0.4], [0.55, 0.45], filename=f4, logger=self.logger)
        output_files.append(f4)

        # Test density plot
        f5 = plot_spatial_temporal_density(
            self.sindex_tracker,
            self.tindex_tracker,
            save_dir=self.plots_dir,
            filename="integration_density",
            logger=self.logger,
        )
        if f5:
            output_files.append(f5)

        for f in output_files:
            self.assertTrue(os.path.exists(f), f"File {f} was not created")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDiagnostics))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
