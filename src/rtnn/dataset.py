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
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Any
import random


class DataPreprocessor(Dataset):
    """
    Dataset class for preprocessing LSM (Land Surface Model) data.

    This class handles loading and preprocessing of NetCDF files containing
    climate data, with support for multiple years, spatial and temporal batching,
    and various normalization techniques.

    Parameters
    ----------
    logger : object
        Logger instance for logging messages.
    dfs : List[str]
        List of file paths to NetCDF files.
    stime : int
        Start time index.
    tstep : int
        Number of time steps per file.
    tbatch : int
        Temporal batch size.
    norm_mapping : Dict, optional
        Dictionary containing normalization statistics for each variable.
        Default is empty dict.
    normalization_type : Dict, optional
        Dictionary specifying normalization type for each variable.
        Default is empty dict.

    Attributes
    ----------
    logger : object
        Logger instance.
    stime : int
        Start time index.
    tstep : int
        Time steps per file.
    tbatch : int
        Temporal batch size.
    norm_mapping : Dict
        Normalization statistics.
    normalization_type : Dict
        Normalization types per variable.
    sbatch : int
        Number of spatial batches.
    years : List[int]
        Sorted list of years in the dataset.
    etime : int
        End time index.
    dfs : List[Tuple[int, int, str]]
        List of (year, spatial_index, file_path) tuples.
    time_blocks : np.ndarray
        Shuffled time blocks.
    min_dims : Dict[str, int]
        Minimum dimensions across files.
    cosz : List[str]
        Cosine of solar zenith angle variable names.
    lai : List[str]
        Leaf area index variable names.
    ssa : List[str]
        Single scattering albedo variable names.
    rs : List[str]
        Surface reflectance variable names.
    ov : List[str]
        Output variable names.

    Examples
    --------
    >>> from rtnn.logger import Logger
    >>> logger = Logger()
    >>> files = ["data_1995.nc", "data_1996.nc"]
    >>> dataset = DataPreprocessor(
    ...     logger=logger,
    ...     dfs=files,
    ...     stime=0,
    ...     tstep=100,
    ...     tbatch=24,
    ...     norm_mapping={},
    ...     normalization_type={}
    ... )
    >>> len(dataset)
    100
    >>> features, targets = dataset[0]
    >>> features.shape
    torch.Size([schunk, feature_channels, seq_length])
    >>> targets.shape
    torch.Size([schunk, output_channels, seq_length])
    """

    def __init__(
        self,
        logger: Any,
        dfs: List[str],
        stime: int,
        tbatch: int,
        training: bool = True,
        sblock_perc: float = 0.6,
        norm_mapping: Dict = {},
        normalization_type: Dict = {},
        debug: bool = False,
    ) -> None:
        """
        Initialize the DataPreprocessor.

        Parameters
        ----------
        logger : Any
            Logger instance for logging messages.
        dfs : List[str]
            List of file paths to NetCDF files.
        stime : int
            Start time index.
        tbatch : int
            Temporal batch size.
        training : bool, optional
            If True, use 60% of spatial batches (data augmentation).
            If False, use 100% of spatial batches (full evaluation).
        norm_mapping : Dict, optional
            Dictionary containing normalization statistics for each variable.
        normalization_type : Dict, optional
            Dictionary specifying normalization type for each variable.
        debug : bool, optional
            If True, print debug information.
        """
        self.logger = logger
        self.stime = stime
        self.tbatch = tbatch
        self.training = training
        self.norm_mapping = norm_mapping
        self.normalization_type = normalization_type
        self.debug = debug
        self.sblock_perc = sblock_perc

        # Group files by year
        self.train_sbatch_files_by_year = defaultdict(list)
        for f in dfs:
            match = re.search(r"_(\d{4})\.nc$", f)
            if match:
                year = int(match.group(1))
                self.train_sbatch_files_by_year[year].append(f)

        # Determine number of spatial batches
        first_key = list(self.train_sbatch_files_by_year.keys())[0]
        self.total_sbatch = len(self.train_sbatch_files_by_year[first_key])

        # Set spatial batch size based on training mode
        if self.training:
            # Training: use 60% of spatial batches
            self.sbatch = max(1, int(self.total_sbatch * self.sblock_perc))
            # Initialize tracking for random spatial mapping
            self.last_tindex = -1
            self.current_spatial_mapping = None
        else:
            # Validation/Testing: use 100% of spatial batches
            self.sbatch = self.total_sbatch

        self.years = sorted(self.train_sbatch_files_by_year.keys())
        self.year_to_index = {y: i for i, y in enumerate(self.years)}

        # Create list of (year, spatial_index, path) for all files
        self.dfs = [
            (year, sindex, path)
            for year in self.years
            for sindex, path in enumerate(sorted(self.train_sbatch_files_by_year[year]))
        ]

        # Find minimum dimensions across all files
        self.min_dims = {
            "time": np.inf,
            "dim_1": np.inf,
            "dim_2": np.inf,
            "dim_3": np.inf,
            "dim_4": np.inf,
        }

        for _, _, file_path in self.dfs:
            ds = xr.open_dataset(file_path)
            for dim in self.min_dims:
                if dim in ds.sizes:
                    self.min_dims[dim] = min(self.min_dims[dim], ds.sizes[dim])
            ds.close()

        for dim, size in self.min_dims.items():
            self.logger.info(f"Minimum {dim} across files: {size}")

        self.tstep = self.min_dims["time"]
        self.etime = self.tstep * len(self.years)

        # Create and shuffle time blocks
        self.time_blocks = np.arange((self.etime - self.stime) // self.tbatch)

        # Define variable groups
        self.cosz = ["coszang"]  # Cosine of solar zenith angle
        self.lai = ["laieff_collim", "laieff_isotrop"]  # Leaf area index
        self.ssa = ["leaf_ssa", "leaf_psd"]  # Single scattering albedo
        self.rs = ["rs_surface_emu"]  # Surface reflectance
        self.ov = [
            "collim_alb",
            "collim_tran",
            "isotrop_alb",
            "isotrop_tran",
        ]  # Output variables

        self.sindex_tracker = []  # Will store spatial indices
        self.tindex_tracker = []  # Will store temporal indices

        self.logger.info(f"Time range: {self.stime} ... {self.etime}")
        self.logger.info(f"Time steps per file: {self.tstep}")
        self.logger.info(f"Temporal batch size: {self.tbatch}")
        self.logger.info(f"Spatial batche size: {self.sbatch}")
        self.logger.info(f"Time blocks: {self.time_blocks}")
        self.logger.info(f"Years: {self.years}")
        self.logger.info(f"Year to index: {self.year_to_index}")
        self.logger.info(
            f"Variable groups: {self.cosz}, {self.lai}, {self.ssa}, {self.rs}, {self.ov}"
        )
        self.logger.info(
            "The list of file info:\n"
            + "\n".join(f"{year}, {sindex}, {path}" for year, sindex, path in self.dfs)
        )
        random.seed(42)  # Set a fixed seed for reproducibility

    def _get_random_spatial_mapping(self) -> List[int]:
        """
        Generate a random spatial mapping for training.

        Returns
        -------
        List[int]
            List of randomly selected processor ranks (size = self.sbatch).
        """
        return random.sample(range(self.total_sbatch), self.sbatch)

    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Normalize data using the specified normalization method.

        Parameters
        ----------
        data : np.ndarray
            Input data array to normalize.
        var_name : str
            Name of the variable for which to retrieve normalization statistics.

        Returns
        -------
        np.ndarray
            Normalized data array.

        Raises
        ------
        ValueError
            If the normalization type is not supported.

        Notes
        -----
        Supported normalization types:
        - minmax: (x - min) / (max - min)
        - standard: (x - mean) / std
        - robust: (x - median) / IQR
        - log1p_minmax: log1p(x) normalized
        - log1p_standard: log1p(x) standardized
        - log1p_robust: log1p(x) robust normalized
        - sqrt_minmax: sqrt(x) normalized
        - sqrt_standard: sqrt(x) standardized
        - sqrt_robust: sqrt(x) robust normalized
        """
        norm_type = self.normalization_type.get(var_name, "log1p_minmax")
        stats = self.norm_mapping[var_name]
        if self.debug:
            self.logger.info(
                f"Normalizing variable '{var_name}' using method '{norm_type}' with stats: {stats}"
            )

        if norm_type == "minmax":
            vmin = stats["vmin"]
            vmax = stats["vmax"]
            return (data - vmin) / (vmax - vmin)

        elif norm_type == "standard":
            mean = stats["vmean"]
            std = stats["vstd"]
            return (data - mean) / std

        elif norm_type == "robust":
            median = stats["median"]
            iqr = stats["iqr"]
            return (data - median) / iqr

        elif norm_type == "log1p_minmax":
            data = np.log1p(data)
            log_min = stats["log_min"]
            log_max = stats["log_max"]
            return (data - log_min) / (log_max - log_min)

        elif norm_type == "log1p_standard":
            data = np.log1p(data)
            mean = stats["log_mean"]
            std = stats["log_std"]
            return (data - mean) / std

        elif norm_type == "log1p_robust":
            data = np.log1p(data)
            median = stats["log_median"]
            iqr = stats["log_iqr"]
            return (data - median) / iqr

        elif norm_type == "sqrt_minmax":
            data = np.sqrt(np.clip(data, a_min=0, a_max=None))
            sqrt_min = stats["sqrt_min"]
            sqrt_max = stats["sqrt_max"]
            return (data - sqrt_min) / (sqrt_max - sqrt_min)

        elif norm_type == "sqrt_standard":
            data = np.sqrt(np.clip(data, a_min=0, a_max=None))
            mean = stats["sqrt_mean"]
            std = stats["sqrt_std"]
            return (data - mean) / std

        elif norm_type == "sqrt_robust":
            data = np.sqrt(np.clip(data, a_min=0, a_max=None))
            median = stats["sqrt_median"]
            iqr = stats["sqrt_iqr"]
            return (data - median) / iqr

        else:
            raise ValueError(
                f"Unsupported normalization type '{norm_type}' for variable '{var_name}'"
            )

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples (time blocks * spatial batches).
        """
        return (self.etime - self.stime) // self.tbatch * self.sbatch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - features: Input features tensor of shape (schunk, feature_channels, seq_length)
            - targets: Target variables tensor of shape (schunk, output_channels, seq_length)

        Notes
        -----
        The method loads data from the appropriate file based on the index,
        applies normalization, and returns the processed features and targets.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        # Calculate spatial and temporal indices
        index_spatial_mapping = index % self.sbatch
        tblock = index // self.sbatch

        # Calculate which year this block belongs to
        blocks_per_year = self.tstep // self.tbatch
        if blocks_per_year <= 0:
            raise ValueError(
                f"Invalid blocks_per_year: {blocks_per_year}. "
                f"tstep={self.tstep}, tbatch={self.tbatch}"
            )

        year_index = tblock // blocks_per_year
        # Validate year_index
        if year_index >= len(self.years):
            raise IndexError(
                f"Year index {year_index} out of range [0, {len(self.years)})"
            )

        local_tblock = tblock % blocks_per_year

        # Calculate time index (with random offset for training)
        tindex = local_tblock * self.tbatch + self.stime

        # For training: regenerate spatial mapping when time index changes
        if self.training:
            if self.last_tindex != tindex:
                self.current_spatial_mapping = self._get_random_spatial_mapping()
                self.last_tindex = tindex
                if self.debug:
                    self.logger.info(
                        f"New spatial mapping for tindex {tindex}: {self.current_spatial_mapping}"
                    )

            # Map the spatial index to an actual processor rank
            sindex = self.current_spatial_mapping[index_spatial_mapping]
        else:
            # For validation/testing: use direct mapping (sindex = index_spatial_mapping)
            sindex = index_spatial_mapping

        if self.training:
            tindex += np.random.randint(self.tbatch)

        self.tindex_tracker.append(tblock)
        self.sindex_tracker.append(sindex)

        # Get the file path
        dfs_index = year_index * self.sbatch + sindex
        _, _, path = self.dfs[dfs_index]

        if self.debug:
            self.logger.info("------------------- GET ITEM INFO -------------------")
            self.logger.info(
                f"\nTorch batch index: {index}\n"
                f"Spatial index before mapping: {index_spatial_mapping}, and Spatial index after mapping: {sindex}\n"
                f"Temporal block index: {tblock}\n"
                f"Year index: {year_index}\n"
                f"Local time block: {local_tblock}\n"
                f"Time index: {tindex}\n"
                f"Loading file: {path}"
            )
        # Open the dataset
        self.df = xr.open_dataset(path)

        # Get dimensions
        # sequence_length_dim = self.min_dims["dim_2"]
        # dim_1 = self.min_dims["dim_1"]
        # dim_3 = self.min_dims["dim_3"]
        # dim_4 = self.min_dims["dim_4"]

        # Get dimensions
        seq_len = self.min_dims["dim_2"]  # 10 (vertical levels)
        dim_1 = self.min_dims["dim_1"]  # 263 (spatial points)
        n_pft = self.min_dims["dim_3"]  # 15
        n_bands = self.min_dims["dim_4"]  # 2

        # self.schunk = dim_1 * dim_3 * dim_4

        # Initialize arrays for each variable group
        # npcosz = np.zeros([self.schunk, len(self.cosz), sequence_length_dim])
        # nplai = np.zeros([self.schunk, len(self.lai), sequence_length_dim])
        # npssa = np.zeros([self.schunk, len(self.ssa), sequence_length_dim])
        # npov = np.zeros([self.schunk, len(self.ov), sequence_length_dim])
        # nprs = np.zeros([self.schunk, len(self.rs), sequence_length_dim])

        # if self.debug:
        #    self.logger.info(
        #        f"Dimensions for processing:\n"
        #        f"  |- sequence_length_dim: {sequence_length_dim}\n"
        #        f"  |- dim_1: {dim_1}\n"
        #        f"  |- dim_3: {dim_3}\n"
        #        f"  |- dim_4: {dim_4}\n"
        #        f"  |- schunk (total spatial chunk size): {self.schunk}"
        #    )
        #    self.logger.info(
        #        f"Initialized numpy arrays for variable groups with shapes:\n"
        #        f"  |- npcosz: {npcosz.shape}\n"
        #        f"  |- nplai: {nplai.shape}\n"
        #        f"  |- npssa: {npssa.shape}\n"
        #        f"  |- npov: {npov.shape}\n"
        #        f"  |- nprs: {nprs.shape}"
        #    )

        # ================================================================
        # FEATURES
        # ================================================================
        # Feature channels:
        # - cosz: 1
        # - lai: 2 vars × n_pft = 30
        # - ssa: 2 vars × n_bands × n_pft = 60
        # - rs: 1 var ×  n_bands × n_pft = 2
        # Total: 93

        n_lai_features = 2 * n_pft  # 30
        n_ssa_features = 2 * n_bands * n_pft  # 60
        n_rs_features = 1 * n_bands * n_pft  # 30
        n_features = 1 + n_lai_features + n_ssa_features + n_rs_features  # 121

        features = np.zeros([dim_1, n_features, seq_len], dtype=np.float32)
        f_idx = 0

        # 1. COSZ - shape: (time, dim_1) -> (dim_1, 1, seq_len)
        for var_name in self.cosz:
            da = self.df[var_name]
            temp = da.isel(time=tindex, dim_1=slice(0, dim_1)).values  # (dim_1,)
            temp = self.normalize(temp, var_name)
            # Tile to (dim_1, 1, seq_len) - same value for all vertical levels
            temp = temp[:, np.newaxis, np.newaxis]  # (dim_1, 1, 1)
            temp = np.tile(temp, (1, 1, seq_len))  # (dim_1, 1, seq_len)
            features[:, f_idx : f_idx + 1, :] = temp
            f_idx += 1

        # 2. LAI - shape: (time, dim_3, dim_2, dim_1) -> (dim_1, n_pft, seq_len)
        # dim_2 is vertical level, we need ALL levels, so we loop over dim_2
        for var_name in self.lai:
            da = self.df[var_name]
            # Get all data for this time step
            temp = da.isel(
                time=tindex, dim_1=slice(0, dim_1)
            ).values  # (dim_3, dim_2, dim_1)
            temp = self.normalize(temp, var_name)
            # For each vertical level (seq_len), we have a (dim_3, dim_1) matrix
            # We want: (dim_1, dim_3, seq_len)
            # Transpose to (dim_1, dim_3, dim_2)
            temp = temp.transpose(2, 0, 1)  # (dim_1, dim_3, dim_2)
            # Now temp has shape (dim_1, n_pft, seq_len) - perfect!
            features[:, f_idx : f_idx + n_pft, :] = temp
            f_idx += n_pft

        # 3. SSA (leaf_ssa, leaf_psd) - shape: (time, dim_4, dim_3) -> (dim_1, n_bands, n_pft, seq_len)
        # Note: SSA does NOT depend on dim_2 (vertical level), so same for all levels
        for var_name in self.ssa:
            da = self.df[var_name]
            temp = da.isel(time=tindex).values  # (dim_4, dim_3)
            temp = self.normalize(temp, var_name)
            # Expand to (dim_1, dim_4, dim_3, seq_len) by tiling
            temp = temp[np.newaxis, :, :, np.newaxis]  # (1, dim_4, dim_3, 1)
            temp = np.tile(
                temp, (dim_1, 1, 1, seq_len)
            )  # (dim_1, dim_4, dim_3, seq_len)
            # Reshape to (dim_1, dim_4 * dim_3, seq_len)
            temp = temp.reshape(dim_1, n_bands * n_pft, seq_len)
            features[:, f_idx : f_idx + n_bands * n_pft, :] = temp
            f_idx += n_bands * n_pft

        # 4. RS (rs_surface_emu) - shape: (time, dim_4, dim_3, dim_1) -> (dim_1, n_bands, n_pft, seq_len)
        # Note: RS has dim_3 (PFT) dimension, we need all PFTs
        for var_name in self.rs:
            da = self.df[var_name]
            temp = da.isel(
                time=tindex, dim_1=slice(0, dim_1)
            ).values  # (dim_4, dim_3, dim_1)
            temp = self.normalize(temp, var_name)
            # Transpose to (dim_1, dim_4, dim_3)
            temp = temp.transpose(2, 0, 1)  # (dim_1, dim_4, dim_3)
            # Expand to (dim_1, dim_4, dim_3, seq_len) by tiling (same for all vertical levels)
            temp = temp[:, :, :, np.newaxis]  # (dim_1, dim_4, dim_3, 1)
            temp = np.tile(temp, (1, 1, 1, seq_len))  # (dim_1, dim_4, dim_3, seq_len)
            # Reshape to (dim_1, dim_4 * dim_3, seq_len)
            temp = temp.reshape(dim_1, n_bands * n_pft, seq_len)
            features[:, f_idx : f_idx + n_bands * n_pft, :] = temp
            f_idx += n_bands * n_pft

        assert f_idx == n_features, f"Feature mismatch: {f_idx} vs {n_features}"

        # ================================================================
        # OUTPUTS - shape: (dim_1, n_outputs, seq_len)
        # n_outputs = 4 vars × n_bands × n_pft = 120
        # ================================================================
        n_outputs = len(self.ov) * n_bands * n_pft  # 120
        outputs = np.zeros([dim_1, n_outputs, seq_len], dtype=np.float32)
        o_idx = 0

        for var_name in self.ov:
            da = self.df[var_name]
            # Shape: (time, dim_4, dim_2, dim_3, dim_1)
            temp = da.isel(
                time=tindex, dim_1=slice(0, dim_1)
            ).values  # (dim_4, dim_2, dim_3, dim_1)
            temp = self.normalize(temp, var_name)
            # We want: (dim_1, dim_4, dim_3, dim_2) - output per vertical level
            # Transpose to (dim_1, dim_4, dim_3, dim_2)
            temp = temp.transpose(3, 0, 2, 1)  # (dim_1, dim_4, dim_3, dim_2)
            # Now temp has shape (dim_1, n_bands, n_pft, seq_len) - perfect!
            # Reshape to (dim_1, n_bands * n_pft, seq_len)
            temp = temp.reshape(dim_1, n_bands * n_pft, seq_len)
            outputs[:, o_idx : o_idx + n_bands * n_pft, :] = temp
            o_idx += n_bands * n_pft

        assert o_idx == n_outputs, f"Output mismatch: {o_idx} vs {n_outputs}"

        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

        return features_tensor, outputs_tensor
