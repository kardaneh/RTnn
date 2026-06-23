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


class RRTMGPDataPreprocessor(Dataset):
    """
    Dataset class for preprocessing RRTMGP radiative transfer data.

    Prepares data in RNN-style format (sequential per layer) which can be:
    - Used directly with RNN models (keeps sequence dimension)
    - Flattened for FNN models using .view(batch, -1)

    For each experiment (expt), returns ALL sites (5120 atmospheric columns).

    Parameters
    ----------
    path : str
        Path to NetCDF file
    training : bool
        If True, enable data augmentation (random sampling of sites)
    norm_mapping : Dict, optional
        Normalization statistics for each variable
    normalization_type : Dict, optional
        Normalization method for each variable
    debug : bool, optional
        self.logger.info debug information
    sblock_perc : float, optional
        Percentage of sites to use for training (default: 0.6)
    """

    def __init__(
        self,
        logger: Any,
        path: str,
        training: bool = True,
        norm_mapping: Dict = {},
        normalization_type: Dict = {},
        debug: bool = False,
        sblock_perc: float = 0.6,
    ) -> None:
        super().__init__()

        self.logger = logger
        self.path = path
        self.training = training
        self.norm_mapping = norm_mapping
        self.normalization_type = normalization_type
        self.debug = debug
        self.sblock_perc = sblock_perc
        random.seed(42)
        # Load the dataset
        if self.debug:
            self.logger.info(f"Loading dataset: {path}")
        self.ds = xr.open_dataset(path)

        # Get dimensions
        self.n_expt = self.ds.sizes["expt"]  # 64
        self.n_site = self.ds.sizes["site"]  # 5120
        self.n_layer = self.ds.sizes["layer"]  # 60
        self.n_level = self.ds.sizes["level"]  # 61
        self.n_gpt = self.ds.sizes["gpt"]  # 224
        self.n_feature = self.ds.sizes["feature"]  # 7

        # Total number of experiments (time steps)
        self.n_experiments = self.n_expt

        # Determine number of sites to use based on training mode
        if self.training:
            # Training: use percentage of sites (data augmentation)
            self.n_sites_used = max(1, int(self.n_site * self.sblock_perc))
            # Initialize tracking for random site mapping
            self.last_expt_idx = -1
            self.current_site_mapping = None
        else:
            # Validation/Testing: use all sites
            self.n_sites_used = self.n_site

        # Data dimensions for RNN-style preparation
        # Features per layer: 11 variables (T, P, H2O, O3, CO2, N2O, CH4, LWP, IWP, mu0, broadband albedo)
        self.n_features_per_layer = 11
        # Auxiliary features: 2 (mu0, broadband albedo)
        self.n_aux_features = 2
        # Outputs: 2 fluxes (rsd, rsu) at each of 61 levels
        self.n_output_fluxes = 2

        # Pre-load data references for faster access
        self.rrtmgp_input = self.ds["rrtmgp_sw_input"]
        self.cloud_lwp = self.ds["cloud_lwp"]
        self.cloud_iwp = self.ds["cloud_iwp"]
        self.mu0 = self.ds["mu0"]
        self.sfc_alb = self.ds["sfc_alb"]
        self.rsd = self.ds["rsd"]
        self.rsu = self.ds["rsu"]

        # Variable names for normalization
        self.gas_vars = ["tlay", "play", "h2o", "o3", "co2", "n2o", "ch4"]
        self.cloud_vars = ["cloud_lwp", "cloud_iwp"]
        self.aux_vars = ["mu0", "sfc_alb"]
        self.flux_vars = ["rsd", "rsu"]

        self.feature_names = self.gas_vars + self.cloud_vars + self.aux_vars
        # ['tlay', 'play', 'h2o', 'o3', 'co2', 'n2o', 'ch4', 'cloud_lwp', 'cloud_iwp', 'mu0', 'sfc_alb']

        self._logger_info()
        self.sindex_tracker = []  # Will store spatial indices
        self.tindex_tracker = []  # Will store temporal indices

    def _get_random_site_mapping(self) -> List[int]:
        """
        Generate a random site mapping for training.

        Returns
        -------
        List[int]
            List of randomly selected site indices (size = n_sites_used)
        """
        return random.sample(range(self.n_site), self.n_sites_used)

    def _logger_info(self):
        """Print dataset information."""
        self.logger.info("=" * 70)
        self.logger.info("RRTMGP DataPreprocessor")
        self.logger.info(f"File: {self.path.split('/')[-1]}")
        self.logger.info(f"Training mode: {self.training}")
        self.logger.info(f"Total experiments (expt): {self.n_experiments}")
        self.logger.info(f"Total sites per experiment: {self.n_site}")
        self.logger.info(
            f"Sites used per experiment: {self.n_sites_used} ({self.sblock_perc*100:.1f}%)"
        )
        self.logger.info(
            f"Dimensions: expt={self.n_expt}, site={self.n_site}, layer={self.n_layer}, level={self.n_level}"
        )
        self.logger.info(f"Features per layer: {self.n_features_per_layer}")
        self.logger.info(f"Auxiliary features: {self.n_aux_features}")
        self.logger.info(
            f"Outputs: {self.n_output_fluxes} fluxes × {self.n_level} levels = {self.n_output_fluxes * self.n_level}"
        )
        self.logger.info("=" * 70)

    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Normalize data using stored statistics.

        Parameters
        ----------
        data : np.ndarray
            Input data to normalize
        var_name : str
            Variable name for lookup

        Returns
        -------
        np.ndarray
            Normalized data
        """
        if not self.norm_mapping or var_name not in self.norm_mapping:
            return data

        norm_type = self.normalization_type.get(
            var_name, "log1p_minmax"
        )  # Default normalization type
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

    def _get_pressure_levels(
        self, expt_idx: int, site_indices: List[int]
    ) -> np.ndarray:
        """
        Extract pressure at level interfaces by averaging adjacent layer pressures.

        For 60 layers (0-59), we get 59 interior levels (1-59).
        Level 0 (TOA) and Level 60 (Surface) are not available from averaging.

        Parameters
        ----------
        expt_idx : int
            Experiment index
        site_indices : List[int]
            List of site indices to extract

        Returns
        -------
        np.ndarray
            Pressure at level interfaces (interior levels 1-59).
            Shape: (n_sites, n_level-2) = (n_sites, 59)
            In Pa.
        """
        # Pressure at layer centers (feature index 1 = play)
        # Shape: (n_sites, n_layer) = (n_sites, 60)
        p_layer = self.rrtmgp_input[expt_idx, site_indices, :, 1].values

        # Average adjacent layers to get pressure at interfaces
        # pres_lev[i] = (p_layer[i] + p_layer[i+1]) / 2
        # This gives levels 1 through 59 (59 levels)
        p_level = (p_layer[:, :-1] + p_layer[:, 1:]) / 2.0  # (n_sites, 59)

        return p_level

    def _get_per_layer_features(
        self, expt_idx: int, site_indices: List[int]
    ) -> np.ndarray:
        """
        Extract per-layer features for multiple sites.

        Parameters
        ----------
        expt_idx : int
            Experiment index
        site_indices : List[int]
            List of site indices to extract

        Returns
        -------
        np.ndarray
            Shape: (n_sites, n_layer, n_features_per_layer) -> (n_sites_used, 60, 11)
        """
        # Gas features: (n_sites, layer, 7)
        gas_data = self.rrtmgp_input[expt_idx, site_indices].values  # (n_sites, 60, 7)

        # Cloud features: (n_sites, layer, 2)
        lwp = self.cloud_lwp[expt_idx, site_indices].values  # (n_sites, 60)
        iwp = self.cloud_iwp[expt_idx, site_indices].values  # (n_sites, 60)
        cloud_data = np.stack([lwp, iwp], axis=-1)  # (n_sites, 60, 2)

        # mu0: (n_sites,) -> tile to (n_sites, n_layer, 1)
        mu0_vals = self.mu0[expt_idx, site_indices].values  # (n_sites,)
        mu0_tiled = np.tile(
            mu0_vals[:, np.newaxis, np.newaxis], (1, self.n_layer, 1)
        )  # (n_sites, 60, 1)

        # Surface albedo: use first g-point (broadband), tile to (n_sites, n_layer, 1)
        albedo_spectral = self.sfc_alb[expt_idx, site_indices].values  # (n_sites, 224)
        albedo_broadband = albedo_spectral[:, 0]  # (n_sites,) - first g-point
        albedo_tiled = np.tile(
            albedo_broadband[:, np.newaxis, np.newaxis], (1, self.n_layer, 1)
        )  # (n_sites, 60, 1)

        # Combine: (n_sites, 60, 11) - [gas(7), cloud(2), mu0(1), albedo(1)]
        layer_features = np.concatenate(
            [
                gas_data,  # (n_sites, 60, 7)
                cloud_data,  # (n_sites, 60, 2)
                mu0_tiled,  # (n_sites, 60, 1)
                albedo_tiled,  # (n_sites, 60, 1)
            ],
            axis=-1,
        )

        return layer_features

    def _get_targets(self, expt_idx: int, site_indices: List[int]) -> np.ndarray:
        """
        Extract target fluxes for multiple sites.

        Parameters
        ----------
        expt_idx : int
            Experiment index
        site_indices : List[int]
            List of site indices to extract

        Returns
        -------
        np.ndarray
            Shape: (n_sites, n_level, n_output_fluxes) -> (n_sites_used, 61, 2)
        """
        rsd_profile = self.rsd[expt_idx, site_indices].values  # (n_sites, 61)
        rsu_profile = self.rsu[expt_idx, site_indices].values  # (n_sites, 61)

        # rsd: remove level 0,
        rsd_selection = rsd_profile[:, 1:]  # (n_sites, 60)
        # rsu: remove last level,
        rsu_selection = rsu_profile[:, :-1]  # (n_sites, 60)

        # Stack along last dimension: (n_sites, 61, 2)
        targets = np.stack([rsd_selection, rsu_selection], axis=-1)

        return targets

    def __len__(self) -> int:
        """Return number of experiments (time steps)."""
        return self.n_experiments

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all sites for a specific experiment.

        Parameters
        ----------
        idx : int
            Experiment index (0 to n_experiments-1)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - layer_features: (n_sites, n_layer, n_features_per_layer) -> (n_sites_used, 60, 9)
            - aux_features: (n_sites, n_aux_features) -> (n_sites_used, 2)
            - targets: (n_sites, n_level, n_output_fluxes) -> (n_sites_used, 61, 2)
        """
        expt_idx = idx

        # Determine site indices based on training mode
        if self.training:
            # For training: regenerate site mapping when experiment changes
            if self.last_expt_idx != expt_idx:
                self.current_site_mapping = self._get_random_site_mapping()
                self.last_expt_idx = expt_idx
                if self.debug:
                    self.logger.info(
                        f"New site mapping for expt {expt_idx}: first 5 sites = {self.current_site_mapping[:5]}"
                    )

            site_indices = self.current_site_mapping
        else:
            # For validation/testing: use all sites
            site_indices = list(range(self.n_site))

        self.sindex_tracker.extend(site_indices)
        self.tindex_tracker.append(expt_idx)
        if self.debug:
            self.logger.info(f"\nExperiment {expt_idx}:")
            self.logger.info(f"  Number of sites: {len(site_indices)}")

        # Extract data for all selected sites
        layer_features = self._get_per_layer_features(
            expt_idx, site_indices
        )  # (n_sites, 60, 11)
        targets = self._get_targets(expt_idx, site_indices)  # (n_sites, 60, 2)
        p_level = self._get_pressure_levels(
            expt_idx, site_indices
        )  # (n_sites, 59) - Pa

        # Normalize features: each of the 11 features separately
        for i, var_name in enumerate(self.feature_names):
            layer_features[..., i] = self.normalize(layer_features[..., i], var_name)

        # Normalize targets: rsd and rsu separately
        targets[..., 0] = self.normalize(targets[..., 0], "rsd")
        targets[..., 1] = self.normalize(targets[..., 1], "rsu")

        # Convert to tensors
        layer_tensor = torch.tensor(layer_features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        p_level_tensor = torch.tensor(p_level, dtype=torch.float32)  # (n_sites, 59)

        # Permute dimensions: (batch, seq, features) -> (batch, features, seq)
        layer_tensor = layer_tensor.permute(0, 2, 1)  # (n_sites, 11, 60)
        targets_tensor = targets_tensor.permute(0, 2, 1)  # (n_sites, 2, 60)

        if self.debug:
            self.logger.info(
                f"  Layer features shape: {layer_tensor.shape}"
            )  # (n_sites, 11, 60)
            self.logger.info(
                f"  Targets shape: {targets_tensor.shape}"
            )  # (n_sites, 2, 60)
            self.logger.info(
                f"  Pressure levels shape: {p_level_tensor.shape}"
            )  # (n_sites, 59)

        return layer_tensor, targets_tensor, p_level_tensor
