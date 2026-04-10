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
import sys
import os
import tempfile
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtnn.dataset import DataPreprocessor
from rtnn.logger import Logger
from rtnn.diagnostics import plot_spatial_temporal_density


def create_dummy_lsm_netcdf(
    filepath: str,
    year: int = 2020,
    processor_rank: int = 0,
    time_steps: int = 100,
    dim_1: int = 10,
    dim_2: int = 10,
    dim_3: int = 15,
    dim_4: int = 2,
) -> None:
    """
    Create a dummy NetCDF file for LSM data testing matching actual data structure.

    Actual data dimensions:
    - time: 1440
    - dim_1: 263 (spatial dimension 1)
    - dim_2: 10 (spatial dimension 2)
    - dim_3: 15 (spatial dimension 3)
    - dim_4: 2 (spatial dimension 4)

    File naming: rtnetcdf_{processor_rank:03d}_{year}.nc

    Parameters
    ----------
    filepath : str
        Path where to save the NetCDF file.
    year : int
        Year for the data.
    processor_rank : int
        Processor rank (0-999).
    time_steps : int
        Number of time steps.
    dim_1, dim_2, dim_3, dim_4 : int
        Dimension sizes matching actual data order.
    """
    import xarray as xr
    import pandas as pd

    # Create coordinates
    time = pd.date_range(f"{year}-01-01", periods=time_steps, freq="D")

    # Create data variables matching actual dimension order
    data_vars = {
        # Input variables
        "coszang": (
            ("time", "dim_1"),
            np.random.randn(time_steps, dim_1) * 0.5 + 0.5,
        ),
        "laieff_collim": (
            ("time", "dim_3", "dim_2", "dim_1"),
            np.random.randn(time_steps, dim_3, dim_2, dim_1) * 0.2 + 0.5,
        ),
        "laieff_isotrop": (
            ("time", "dim_3", "dim_2", "dim_1"),
            np.random.randn(time_steps, dim_3, dim_2, dim_1) * 0.2 + 0.5,
        ),
        "leaf_ssa": (
            ("time", "dim_4", "dim_3"),
            np.random.randn(time_steps, dim_4, dim_3) * 0.1 + 0.3,
        ),
        "leaf_psd": (
            ("time", "dim_4", "dim_3"),
            np.random.randn(time_steps, dim_4, dim_3) * 0.5 + 1.0,
        ),
        "rs_surface_emu": (
            ("time", "dim_4", "dim_3", "dim_1"),
            np.random.randn(time_steps, dim_4, dim_3, dim_1) * 0.1 + 0.2,
        ),
        # Output variables
        "collim_alb": (
            ("time", "dim_4", "dim_2", "dim_3", "dim_1"),
            np.random.randn(time_steps, dim_4, dim_2, dim_3, dim_1) * 0.1 + 0.5,
        ),
        "collim_tran": (
            ("time", "dim_4", "dim_2", "dim_3", "dim_1"),
            np.random.randn(time_steps, dim_4, dim_2, dim_3, dim_1) * 0.1 + 0.3,
        ),
        "isotrop_alb": (
            ("time", "dim_4", "dim_2", "dim_3", "dim_1"),
            np.random.randn(time_steps, dim_4, dim_2, dim_3, dim_1) * 0.1 + 0.5,
        ),
        "isotrop_tran": (
            ("time", "dim_4", "dim_2", "dim_3", "dim_1"),
            np.random.randn(time_steps, dim_4, dim_2, dim_3, dim_1) * 0.1 + 0.3,
        ),
    }

    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time,
            "dim_1": np.arange(dim_1),
            "dim_2": np.arange(dim_2),
            "dim_3": np.arange(dim_3),
            "dim_4": np.arange(dim_4),
        },
    )

    ds.to_netcdf(filepath)
    ds.close()


def create_multiple_years_data(
    temp_dir: str,
    years: list,
    processor_ranks: list = None,
    time_steps: int = 100,
    dim_1: int = 12,
    dim_2: int = 10,
    dim_3: int = 15,
    dim_4: int = 2,
) -> list:
    """
    Create multiple NetCDF files for different years and processor ranks.

    File naming convention: rtnetcdf_{processor_rank:03d}_{year}.nc

    Returns
    -------
    list of str
        List of created file paths.
    """
    if processor_ranks is None:
        processor_ranks = [0, 1, 2]

    files = []
    for year in years:
        for rank in processor_ranks:
            filename = f"rtnetcdf_{rank:03d}_{year}.nc"
            filepath = os.path.join(temp_dir, filename)
            create_dummy_lsm_netcdf(
                filepath,
                year=year,
                processor_rank=rank,
                time_steps=time_steps,
                dim_1=dim_1,
                dim_2=dim_2,
                dim_3=dim_3,
                dim_4=dim_4,
            )
            files.append(filepath)
    return files


def create_norm_mapping() -> dict:
    """
    Create dummy normalization mapping for testing.

    Returns
    -------
    dict
        Normalization mapping dictionary.
    """
    norm_mapping = {}

    for var in [
        "coszang",
        "laieff_collim",
        "laieff_isotrop",
        "leaf_ssa",
        "leaf_psd",
        "rs_surface_emu",
        "collim_alb",
        "collim_tran",
        "isotrop_alb",
        "isotrop_tran",
    ]:
        norm_mapping[var] = {
            "vmin": 0.0,
            "vmax": 1.0,
            "vmean": 0.5,
            "vstd": 0.3,
            "median": 0.5,
            "iqr": 0.4,
            "log_min": -2.0,
            "log_max": 1.0,
            "log_mean": -0.5,
            "log_std": 0.8,
            "log_median": -0.5,
            "log_iqr": 0.9,
            "sqrt_min": 0.0,
            "sqrt_max": 1.0,
            "sqrt_mean": 0.6,
            "sqrt_std": 0.3,
            "sqrt_median": 0.6,
            "sqrt_iqr": 0.4,
        }

    return norm_mapping


def verify_file_naming_convention(filepath: str) -> tuple:
    """
    Verify that a file follows the naming convention and extract year and rank.

    Returns
    -------
    tuple
        (processor_rank, year)
    """
    import re

    basename = os.path.basename(filepath)
    match = re.match(r"rtnetcdf_(\d{3})_(\d{4})\.nc$", basename)
    if not match:
        raise ValueError(
            f"File {basename} does not match naming convention: rtnetcdf_XXX_YYYY.nc"
        )

    processor_rank = int(match.group(1))
    year = int(match.group(2))
    return processor_rank, year


class TestDataPreprocessor(unittest.TestCase):
    """Unit tests for DataPreprocessor class with real NetCDF files."""

    def setUp(self):
        """Set up test fixtures with real data files matching the naming convention."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create dummy NetCDF files for years 1995-1997
        self.years = [1995, 1996, 1997]
        self.processor_ranks = [0, 1, 2, 3, 4]  # Simulate 5 processors
        self.time_steps = 25
        self.dim_1 = 12
        self.dim_2 = 10
        self.dim_3 = 15
        self.dim_4 = 2
        self.spatial_batches = len(self.processor_ranks)

        # Create files following the naming convention
        self.files = create_multiple_years_data(
            temp_dir=self.temp_dir,
            years=self.years,
            processor_ranks=self.processor_ranks,
            time_steps=self.time_steps,
            dim_1=self.dim_1,
            dim_2=self.dim_2,
            dim_3=self.dim_3,
            dim_4=self.dim_4,
        )

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        # Create normalization mapping
        self.norm_mapping = create_norm_mapping()

        self.normalization_type = {
            "coszang": "log1p_standard",
            "laieff_collim": "log1p_standard",
            "laieff_isotrop": "log1p_standard",
            "leaf_ssa": "log1p_standard",
            "leaf_psd": "log1p_standard",
            "rs_surface_emu": "log1p_standard",
            "collim_alb": "log1p_standard",
            "collim_tran": "log1p_standard",
            "isotrop_alb": "log1p_standard",
            "isotrop_tran": "log1p_standard",
        }

        # Create dataset
        self.logger.info("Creating DataPreprocessor instance...")
        self.dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files,
            stime=0,
            tbatch=25,
            norm_mapping=self.norm_mapping,
            normalization_type=self.normalization_type,
            debug=False,
        )

    # ------------------------------------------------------------------------
    # Naming Convention Tests
    # ------------------------------------------------------------------------

    def test_file_naming_convention(self):
        """Test that all generated files follow the naming convention."""
        self.logger.info("Testing file naming convention...")
        for filepath in self.files:
            rank, year = verify_file_naming_convention(filepath)
            self.assertIn(rank, self.processor_ranks)
            self.assertIn(year, self.years)
        self.logger.success("File naming convention test passed")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_initialization(self):
        """Test DataPreprocessor initialization with real files."""
        self.logger.info("Testing dataset initialization...")
        self.assertEqual(self.dataset.stime, 0)
        self.assertEqual(self.dataset.tstep, self.time_steps)
        self.assertEqual(self.dataset.tbatch, 25)
        self.assertEqual(len(self.dataset.years), len(self.years))
        self.assertEqual(self.dataset.sbatch, self.spatial_batches)

        for year in self.years:
            self.assertIn(year, self.dataset.years)

        first_year = self.years[0]
        self.assertEqual(
            len(self.dataset.train_sbatch_files_by_year[first_year]),
            len(self.processor_ranks),
        )
        self.logger.success("Initialization test passed")

    def test_min_dims_calculation(self):
        """Test minimum dimensions are calculated correctly."""
        self.logger.info("Testing minimum dimensions calculation...")
        self.assertIn("dim_1", self.dataset.min_dims)
        self.assertIn("dim_2", self.dataset.min_dims)
        self.assertIn("dim_3", self.dataset.min_dims)
        self.assertIn("dim_4", self.dataset.min_dims)

        self.assertNotEqual(self.dataset.min_dims["dim_1"], np.inf)
        self.assertEqual(self.dataset.min_dims["dim_1"], self.dim_1)
        self.assertEqual(self.dataset.min_dims["dim_2"], self.dim_2)
        self.assertEqual(self.dataset.min_dims["dim_3"], self.dim_3)
        self.assertEqual(self.dataset.min_dims["dim_4"], self.dim_4)
        self.logger.success("Minimum dimensions test passed")

    def test_variable_groups(self):
        """Test variable groups are correctly defined."""
        self.logger.info("Testing variable groups...")
        self.assertEqual(self.dataset.cosz, ["coszang"])
        self.assertEqual(self.dataset.lai, ["laieff_collim", "laieff_isotrop"])
        self.assertEqual(self.dataset.ssa, ["leaf_ssa", "leaf_psd"])
        self.assertEqual(self.dataset.rs, ["rs_surface_emu"])
        self.assertEqual(
            self.dataset.ov,
            ["collim_alb", "collim_tran", "isotrop_alb", "isotrop_tran"],
        )
        self.logger.success("Variable groups test passed")

    # ------------------------------------------------------------------------
    # Length Tests
    # ------------------------------------------------------------------------

    def test_len_calculation(self):
        """Test __len__ method returns correct number of samples."""
        self.logger.info("Testing length calculation...")
        expected_len = (
            (self.time_steps - 0) // 25 * self.spatial_batches * len(self.years)
        )
        self.assertEqual(len(self.dataset), expected_len)
        self.logger.success(f"Length calculation test passed: {expected_len} samples")

    # ------------------------------------------------------------------------
    # Normalization Tests
    # ------------------------------------------------------------------------

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        self.logger.info("Testing min-max normalization...")
        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files[:1],
            stime=0,
            tbatch=25,
            norm_mapping={"test_var": {"vmin": 0, "vmax": 10}},
            normalization_type={"test_var": "minmax"},
            debug=False,
        )

        data = np.array([2.0, 5.0, 8.0])
        normalized = dataset.normalize(data, "test_var")

        expected = np.array([0.2, 0.5, 0.8])
        np.testing.assert_array_almost_equal(normalized, expected)
        self.logger.success("Min-max normalization test passed")

    def test_normalize_standard(self):
        """Test standard normalization."""
        self.logger.info("Testing standard normalization...")
        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files[:1],
            stime=0,
            tbatch=25,
            norm_mapping={"test_var": {"vmean": 5, "vstd": 2}},
            normalization_type={"test_var": "standard"},
            debug=False,
        )

        data = np.array([3.0, 5.0, 7.0])
        normalized = dataset.normalize(data, "test_var")

        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)
        self.logger.success("Standard normalization test passed")

    def test_normalize_log1p_minmax(self):
        """Test log1p min-max normalization."""
        self.logger.info("Testing log1p min-max normalization...")
        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files[:1],
            stime=0,
            tbatch=25,
            norm_mapping={"test_var": {"log_min": 0, "log_max": 1}},
            normalization_type={"test_var": "log1p_minmax"},
            debug=False,
        )

        data = np.array([0, np.exp(1) - 1])
        normalized = dataset.normalize(data, "test_var")

        np.testing.assert_array_almost_equal(normalized, np.array([0.0, 1.0]))
        self.logger.success("Log1p min-max normalization test passed")

    def test_normalize_unsupported_type(self):
        """Test unsupported normalization type raises error."""
        self.logger.info("Testing unsupported normalization type...")
        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files[:1],
            stime=0,
            tbatch=25,
            norm_mapping={"test_var": {"vmin": 0, "vmax": 1}},
            normalization_type={"test_var": "invalid_type"},
            debug=False,
        )

        with self.assertRaises(ValueError):
            dataset.normalize(np.array([1, 2, 3]), "test_var")
        self.logger.success("Unsupported type test passed")

    # ------------------------------------------------------------------------
    # Data Loading Tests
    # ------------------------------------------------------------------------

    def test_getitem_returns_valid_data(self):
        """Test __getitem__ returns valid tensors."""
        self.logger.info("Testing __getitem__ returns valid data...")
        features, targets = self.dataset[0]

        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)

        # feature_channels = len(cosz) + len(lai) + len(ssa) + len(rs) = 1 + 2 + 2 + 1 = 6
        expected_feature_channels = 6
        self.assertEqual(features.dim(), 3)
        self.assertEqual(features.shape[1], expected_feature_channels)

        self.assertEqual(targets.dim(), 3)
        self.assertEqual(targets.shape[1], 4)

        seq_length = self.dataset.min_dims["dim_2"]
        self.assertEqual(features.shape[2], seq_length)
        self.assertEqual(targets.shape[2], seq_length)

        expected_schunk = self.dim_1 * self.dim_3 * self.dim_4
        self.assertEqual(features.shape[0], expected_schunk)
        self.assertEqual(targets.shape[0], expected_schunk)

        self.logger.success("Data loading test passed")

    def test_getitem_different_indices(self):
        """Test __getitem__ works with different indices."""
        self.logger.info("Testing __getitem__ with different indices...")
        indices = [0, len(self.dataset) // 2, len(self.dataset) - 1]

        for idx in indices:
            features, targets = self.dataset[idx]
            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)
            self.assertGreater(features.numel(), 0)
            self.assertGreater(targets.numel(), 0)
            self.assertFalse(torch.isnan(features).any())
            self.assertFalse(torch.isnan(targets).any())

        self.logger.success("Different indices test passed")

    def test_getitem_covers_all_processor_ranks(self):
        """Test that over time, all processor ranks are accessed."""
        self.logger.info("Testing coverage of all processor ranks...")
        all_ranks_accessed = set()
        for idx in range(self.spatial_batches):
            features, targets = self.dataset[idx]
            all_ranks_accessed.add(idx % self.spatial_batches)

        self.assertEqual(len(all_ranks_accessed), self.spatial_batches)
        self.logger.success("Processor ranks coverage test passed")

    def test_getitem_covers_all_years(self):
        """Test that samples from all years are accessible."""
        self.logger.info("Testing coverage of all years...")
        total_samples = len(self.dataset)
        samples_per_year = total_samples // len(self.years)

        for year_idx in range(len(self.years)):
            idx = year_idx * samples_per_year
            features, targets = self.dataset[idx]
            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)

        self.logger.success("Years coverage test passed")

    # ------------------------------------------------------------------------
    # Edge Cases Tests
    # ------------------------------------------------------------------------

    def test_single_file(self):
        """Test with a single file."""
        self.logger.info("Testing single file dataset...")
        single_file = self.files[:1]
        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=single_file,
            stime=0,
            tbatch=25,
            norm_mapping=self.norm_mapping,
            normalization_type=self.normalization_type,
            debug=False,
        )

        self.assertEqual(len(dataset.years), 1)
        features, targets = dataset[0]
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.logger.success("Single file test passed")

    def test_single_processor_rank(self):
        """Test with only one processor rank."""
        self.logger.info("Testing single processor rank dataset...")
        single_rank_temp_dir = tempfile.mkdtemp()
        single_rank_files = create_multiple_years_data(
            temp_dir=single_rank_temp_dir,
            years=self.years[:1],
            processor_ranks=[0],
            time_steps=self.time_steps,
            dim_1=self.dim_1,
            dim_2=self.dim_2,
            dim_3=self.dim_3,
            dim_4=self.dim_4,
        )

        dataset = DataPreprocessor(
            logger=self.logger,
            dfs=single_rank_files,
            stime=0,
            tbatch=25,
            norm_mapping=self.norm_mapping,
            normalization_type=self.normalization_type,
            debug=False,
        )

        self.assertEqual(dataset.sbatch, 1)
        features, targets = dataset[0]
        self.assertIsInstance(features, torch.Tensor)

        shutil.rmtree(single_rank_temp_dir)
        self.logger.success("Single processor rank test passed")

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------

    def test_train_dataloader_integration(self):
        """Test that DataPreprocessor works with PyTorch DataLoader."""
        self.logger.info("Testing DataLoader integration...")
        self.dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files,
            stime=0,
            tbatch=1,
            norm_mapping=self.norm_mapping,
            normalization_type=self.normalization_type,
            debug=True,
        )
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False, num_workers=0)
        loop = tqdm(
            enumerate(loader),
            total=len(loader),
            desc="DataLoader Integration Test",
            unit="batch",
        )

        for _, (features, targets) in loop:
            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)
            self.assertEqual(features.dim(), 4)
            self.assertEqual(targets.dim(), 4)

        plot_spatial_temporal_density(
            sindex_tracker=self.dataset.sindex_tracker,
            tindex_tracker=self.dataset.tindex_tracker,
            mode="train",
            save_dir="./tests_plots",
            filename="density_scatter_test",
        )
        self.logger.success("Train DataLoader integration test passed")

    def test_valid_dataloader_integration(self):
        """Test that DataPreprocessor works with PyTorch DataLoader."""
        self.logger.info("Testing DataLoader integration...")
        self.dataset = DataPreprocessor(
            logger=self.logger,
            dfs=self.files,
            stime=0,
            tbatch=25,
            training=False,
            norm_mapping=self.norm_mapping,
            normalization_type=self.normalization_type,
            debug=True,
        )
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False, num_workers=0)
        loop = tqdm(
            enumerate(loader),
            total=len(loader),
            desc="DataLoader Integration Test",
            unit="batch",
        )

        for _, (features, targets) in loop:
            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)
            self.assertEqual(features.dim(), 4)
            self.assertEqual(targets.dim(), 4)
        self.logger.success("Train DataLoader integration test passed")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
