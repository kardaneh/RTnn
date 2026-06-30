# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Any
import random


class REFTRANSDataPreprocessor(Dataset):
    """
    Dataset class for FNN-RefTrans emulator.

    Inputs (5 features): tau, ssa, g, mu0, Tnoscat
    Outputs (4 features): rdif, tdif, rdir, tdir

    For each experiment (expt), returns data in RNN format (B, C, T):
    - B: n_sites_per_batch * n_gpt (batch dimension)
    - C: n_features = 5 (tau, ssa, g, mu0, Tnoscat)
    - T: n_layer = 60 (sequence length)

    Site selection:
    - Training: Random start index per batch per experiment with periodic boundary
    - Validation: Sequential all sites without wrap

    Parameters
    ----------
    logger : Any
        Logger instance
    path : str
        Path to NetCDF file
    training : bool
        If True, enable data augmentation (random sampling of sites)
    norm_mapping : Dict, optional
        Normalization statistics for each variable
    normalization_type : Dict, optional
        Normalization method for each variable
    debug : bool, optional
        Enable debug logging
    n_sites_per_batch : int, optional
        Number of sites per spatial batch (default: 128)
    sbatch : int, optional
        Number of spatial batches for training (default: 64)
    """

    def __init__(
        self,
        logger: Any,
        path: str,
        training: bool = True,
        norm_mapping: Dict = {},
        normalization_type: Dict = {},
        debug: bool = False,
        n_sites_per_batch: int = 128,
        sbatch: int = 256,
    ) -> None:
        super().__init__()

        self.logger = logger
        self.path = path
        self.training = training
        self.norm_mapping = norm_mapping
        self.normalization_type = normalization_type
        self.debug = debug
        self.n_sites_per_batch = n_sites_per_batch
        self.sbatch = sbatch

        if self.debug:
            self.logger.info(f"Loading dataset: {path}")
        self.ds = xr.open_dataset(path)

        # Get dimensions
        self.n_expt = self.ds.sizes["expt"]  # Usually 1
        self.n_site = self.ds.sizes["site"]  # 32768
        self.n_layer = self.ds.sizes["layer"]  # 60
        self.n_gpt = self.ds.sizes["gpt"]  # 224

        # Total number of experiments
        self.n_experiments = self.n_expt

        # Determine number of spatial batches based on mode
        if self.training:
            # Training: use sbatch (default 64)
            self.n_batches = self.sbatch
            # Total sites used = n_sites_per_batch * sbatch (can exceed n_site due to periodic boundary)
            self.n_sites_used = self.n_sites_per_batch * self.sbatch
            # Initialize tracking for random start indices
            self.last_expt_idx = -1
            self.current_start_indices = None  # List of n_batches random start indices
            if self.debug:
                self.logger.info(
                    f"Training: {self.n_batches} batches, {self.n_sites_used} total sites used (with periodicity)"
                )
        else:
            # Validation: use all sites, determine number of batches
            self.n_batches = max(1, self.n_site // self.n_sites_per_batch)
            # If there are remaining sites, add one more batch
            if self.n_site % self.n_sites_per_batch != 0:
                self.n_batches += 1
            self.n_sites_used = self.n_site
            if self.debug:
                self.logger.info(
                    f"Validation: {self.n_batches} batches, {self.n_sites_used} total sites"
                )

        # Data dimensions for RNN format
        # Input features: tau, ssa, g, mu0, Tnoscat (5 features)
        self.n_features = 5
        # Outputs: rdif, tdif, rdir, tdir (4 outputs)
        self.n_outputs = 4

        # Pre-load data references
        self.mu0 = self.ds["mu0"]

        # Optical properties (inputs)
        self.tau_sw = self.ds["tau_sw"]
        self.ssa_sw = self.ds["ssa_sw"]
        self.g_sw = self.ds["g_sw"]

        # Outputs
        self.rdif = self.ds["rdif"]
        self.tdif = self.ds["tdif"]
        self.rdir = self.ds["rdir"]
        self.tdir = self.ds["tdir"]

        # Feature names for normalization (5 features)
        self.feature_names = ["tau_sw", "ssa_sw", "g_sw", "mu0", "tnoscat"]
        self.output_names = ["rdif", "tdif", "rdir", "tdir"]

        self._logger_info()

    def _get_random_start_indices(self) -> List[int]:
        """
        Generate random start indices for each batch.
        Returns a list of n_batches random integers between 0 and n_site - 1.
        """
        return [random.randint(0, self.n_site - 1) for _ in range(self.n_batches)]

    def _get_site_indices_for_batch(self, start_idx: int, batch_idx: int) -> List[int]:
        """
        Get site indices for a specific batch.

        For training: uses periodic boundary (wrap around with modulo)
        For validation: sequential without wrap, padded if necessary

        Parameters
        ----------
        start_idx : int
            Starting index for this batch
        batch_idx : int
            Batch index (0 to n_batches-1) - used for validation only

        Returns
        -------
        List[int]
            List of site indices for this batch
        """
        if self.training:
            # Training: periodic boundary using modulo
            batch_start = start_idx
            batch_end = start_idx + self.n_sites_per_batch
            site_indices = [i % self.n_site for i in range(batch_start, batch_end)]
            return site_indices
        else:
            # Validation: sequential without wrap
            batch_start = batch_idx * self.n_sites_per_batch
            batch_end = min(batch_start + self.n_sites_per_batch, self.n_site)
            site_indices = list(range(batch_start, batch_end))
            return site_indices

    def _logger_info(self):
        """Log dataset information."""
        self.logger.info("=" * 70)
        self.logger.info("REFTRANS DataPreprocessor (RNN-RefTrans)")
        self.logger.info(f"File: {self.path.split('/')[-1]}")
        self.logger.info(f"Training mode: {self.training}")
        self.logger.info(f"Spatial batches: {self.n_batches}")
        self.logger.info(f"Sites per batch: {self.n_sites_per_batch}")
        self.logger.info(
            f"Total sites used: {self.n_sites_used} (out of {self.n_site})"
        )
        self.logger.info(f"Total experiments (expt): {self.n_experiments}")
        self.logger.info(f"Total sites per experiment: {self.n_site}")
        self.logger.info(
            f"Dimensions: expt={self.n_expt}, site={self.n_site}, layer={self.n_layer}, gpt={self.n_gpt}"
        )
        self.logger.info(f"Features: {self.n_features} (tau, ssa, g, mu0, Tnoscat)")
        self.logger.info(f"Outputs: {self.n_outputs} (rdif, tdif, rdir, tdir)")
        self.logger.info(
            f"RNN format: (B={self.n_sites_per_batch}*{self.n_gpt}, C={self.n_features}, T={self.n_layer})"
        )
        if self.training:
            self.logger.info(
                "Site selection: Periodic boundary with random start index per batch per experiment"
            )
        else:
            self.logger.info("Site selection: Sequential all sites (no wrap)")
        self.logger.info("=" * 70)

    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data using stored statistics."""
        if not self.norm_mapping or var_name not in self.norm_mapping:
            return data

        norm_type = self.normalization_type.get(var_name, "minmax")
        stats = self.norm_mapping[var_name]

        if norm_type == "minmax":
            vmin, vmax = stats["vmin"], stats["vmax"]
            return (data - vmin) / (vmax - vmin + 1e-8)

        elif norm_type == "standard":
            mean, std = stats["vmean"], stats["vstd"]
            return (data - mean) / (std + 1e-8)

        elif norm_type == "robust":
            median, iqr = stats["median"], stats["iqr"]
            return (data - median) / (iqr + 1e-8)

        elif norm_type == "log1p_standard":
            data_log = np.log1p(np.clip(data, a_min=0, a_max=None))
            mean, std = stats["log_mean"], stats["log_std"]
            return (data_log - mean) / (std + 1e-8)

        elif norm_type == "log1p_minmax":
            data_log = np.log1p(np.clip(data, a_min=0, a_max=None))
            vmin, vmax = stats["log_min"], stats["log_max"]
            return (data_log - vmin) / (vmax - vmin + 1e-8)

        elif norm_type == "log1p_robust":
            data_log = np.log1p(np.clip(data, a_min=0, a_max=None))
            median, iqr = stats["log_median"], stats["log_iqr"]
            return (data_log - median) / (iqr + 1e-8)

        elif norm_type == "sqrt_standard":
            data_sqrt = np.sqrt(np.clip(data, a_min=0, a_max=None))
            mean, std = stats["sqrt_mean"], stats["sqrt_std"]
            return (data_sqrt - mean) / (std + 1e-8)

        elif norm_type == "sqrt_minmax":
            data_sqrt = np.sqrt(np.clip(data, a_min=0, a_max=None))
            vmin, vmax = stats["sqrt_min"], stats["sqrt_max"]
            return (data_sqrt - vmin) / (vmax - vmin + 1e-8)

        elif norm_type == "sqrt_robust":
            data_sqrt = np.sqrt(np.clip(data, a_min=0, a_max=None))
            median, iqr = stats["sqrt_median"], stats["sqrt_iqr"]
            return (data_sqrt - median) / (iqr + 1e-8)

        else:
            return data

    def __len__(self) -> int:
        """Return number of samples (experiments * spatial batches)."""
        return self.n_experiments * self.n_batches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a specific spatial batch for a specific experiment.

        Parameters
        ----------
        idx : int
            Index combining experiment and spatial batch

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - features: (n_sites_per_batch * n_gpt, n_features, n_layer) where n_features=5
            - targets: (n_sites_per_batch * n_gpt, n_outputs, n_layer) where n_outputs=4
        """
        batch_idx = idx % self.n_batches
        expt_idx = idx // self.n_batches

        # Determine site indices based on training mode
        if self.training:
            # Regenerate random start indices when experiment changes
            if self.last_expt_idx != expt_idx:
                self.current_start_indices = self._get_random_start_indices()
                self.last_expt_idx = expt_idx
                if self.debug:
                    self.logger.info(
                        f"New start indices for expt {expt_idx}: {self.current_start_indices}"
                    )

            # Get the start index for this specific batch
            start_idx = self.current_start_indices[batch_idx]

            # Get sites for this batch using periodic boundary
            site_indices = self._get_site_indices_for_batch(start_idx, batch_idx)
        else:
            # For validation: sequential batches without wrap
            site_indices = self._get_site_indices_for_batch(0, batch_idx)

        n_sites = len(site_indices)

        if self.debug:
            self.logger.info(f"\nExperiment {expt_idx}, Batch {batch_idx}:")
            self.logger.info(f"  Number of sites: {n_sites}")
            self.logger.info(f"  Sites: {site_indices}")

        # Extract features: tau, ssa, g
        # Shape: (n_sites, n_layer, n_gpt)
        tau = self.tau_sw[expt_idx, site_indices, :, :].values
        ssa = self.ssa_sw[expt_idx, site_indices, :, :].values
        g = self.g_sw[expt_idx, site_indices, :, :].values

        # mu0: (n_sites,) -> expand to (n_sites, n_layer, n_gpt)
        mu0_vals = self.mu0[expt_idx, site_indices].values  # (n_sites,)
        mu0 = np.tile(
            mu0_vals[:, np.newaxis, np.newaxis], (1, self.n_layer, self.n_gpt)
        )

        # Compute Tnoscat = exp(-tau / mu0)
        mu0_safe = np.where(mu0 > 1e-8, mu0, 1e-8)
        tnoscat = np.exp(-tau / mu0_safe)

        # Stack features: (n_sites, n_layer, n_gpt, 5)
        features = np.stack([tau, ssa, g, mu0, tnoscat], axis=-1)

        # Extract targets: rdif, tdif, rdir, tdir
        # Shape: (n_sites, n_layer, n_gpt)
        rdif = self.rdif[expt_idx, site_indices, :, :].values
        tdif = self.tdif[expt_idx, site_indices, :, :].values
        rdir = self.rdir[expt_idx, site_indices, :, :].values
        tdir = self.tdir[expt_idx, site_indices, :, :].values

        # Stack targets: (n_sites, n_layer, n_gpt, 4)
        targets = np.stack([rdif, tdif, rdir, tdir], axis=-1)

        # Normalize features: apply to last dimension
        for i, var_name in enumerate(self.feature_names):
            features[..., i] = self.normalize(features[..., i], var_name)

        # Normalize targets: apply to last dimension
        for i, var_name in enumerate(self.output_names):
            targets[..., i] = self.normalize(targets[..., i], var_name)

        # Reshape to (n_sites * n_gpt, n_layer, n_features)
        features = features.reshape(-1, self.n_layer, self.n_features)
        targets = targets.reshape(-1, self.n_layer, self.n_outputs)

        # Permute to (n_sites * n_gpt, n_features, n_layer)
        features = np.transpose(features, (0, 2, 1))
        targets = np.transpose(targets, (0, 2, 1))

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        if self.debug:
            self.logger.info(
                f"  Features shape (B, C, T): {features_tensor.shape}"
            )  # (n_sites*n_gpt, 5, 60)
            self.logger.info(
                f"  Targets shape (B, C, T): {targets_tensor.shape}"
            )  # (n_sites*n_gpt, 4, 60)

        return features_tensor, targets_tensor
