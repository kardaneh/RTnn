import torch
from torch.utils.data import Dataset
import os
import numpy as np
import xarray as xr
from collections import defaultdict
import re

class DataPreprocessor(Dataset):
    """
    """
    def __init__(
            self, 
            logger, 
            dfs, 
            stime, 
            tstep, 
            tbatch,
            norm_mapping={},
            normalization_type={}
            ):
        self.logger = logger
        self.stime = stime
        self.tstep = tstep
        self.tbatch = tbatch
        self.norm_mapping = norm_mapping
        self.normalization_type = normalization_type


        self.train_sbatch_files_by_year = defaultdict(list)
        for f in dfs:
            match = re.search(r'_(\d{4})\.nc$', f)
            if match:
                year = int(match.group(1))
                self.train_sbatch_files_by_year[year].append(f)

        first_key = list(self.train_sbatch_files_by_year.keys())[0]
        self.sbatch = len(self.train_sbatch_files_by_year[first_key])
        self.years = sorted(self.train_sbatch_files_by_year.keys())
        self.year_to_index = {y: i for i, y in enumerate(self.years)}
        self.etime = self.tstep * len(self.years)

        self.dfs = [(year, sindex, path)
                for year in self.years
                for sindex, path in enumerate(sorted(self.train_sbatch_files_by_year[year]))
                ]
        

        self.time_blocks = np.arange((self.etime - self.stime) // self.tbatch)
        np.random.shuffle(self.time_blocks)

        self.logger.info(f"Time range: {self.stime} ... {self.etime}")
        self.logger.info(f"Spatial batches: {self.sbatch}")
        self.logger.info(f"Temporal batches: {self.tbatch}")
        
        self.min_dims = {
                'time': np.inf,
                'dim_1': np.inf,
                'dim_2': np.inf,
                'dim_3': np.inf,
                'dim_4': np.inf
                }
        
        for _, _, file_path in self.dfs:
            ds = xr.open_dataset(file_path, engine="netcdf4")
            for dim in self.min_dims:
                if dim in ds.sizes:
                    self.min_dims[dim] = min(self.min_dims[dim], ds.sizes[dim])
            ds.close()

        self.cosz = [
                'coszang'
                ]
        self.lai = [
                'laieff_collim',
                'laieff_isotrop'
                ]
        self.ssa = [
                'leaf_ssa',
                'leaf_psd'
                ]
        self.rs = [
                'rs_surface_emu'
                ]

        self.ov = [
                'collim_alb', 
                'collim_tran', 
                'isotrop_alb', 
                'isotrop_tran'
                ]

    def shuffle_time_blocks(self):
        np.random.shuffle(self.time_blocks)

    def normalize(self, data, var_name):
        norm_type = self.normalization_type.get(var_name, "log1p_minmax")
        stats = self.norm_mapping[var_name]

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
            raise ValueError(f"Unsupported normalization type '{norm_type}' for variable '{var_name}'")

    def __len__(self):
        return (self.etime - self.stime) // self.tbatch * self.sbatch

    def __getitem__(self, index):
        """
        """
        #tindex = (index // self.sbatch) * self.tbatch + self.stime + np.random.randint(self.tbatch)
        #sindex =  index % self.sbatch
        #tblock_index = index // self.sbatch
        #tblock = self.time_blocks[tblock_index]
        #tindex = tblock * self.tbatch + self.stime + np.random.randint(self.tbatch)
        #self.df = self.df = xr.open_dataset(self.dfs[sindex], engine="netcdf4") #self.loaded_dfs[sindex]
        
        sindex = index % self.sbatch
        tblock_index = index // self.sbatch
        tblock = self.time_blocks[tblock_index]

        year_index = tblock // (self.tstep // self.tbatch)
        local_tblock = tblock % (self.tstep // self.tbatch)
        tindex = local_tblock * self.tbatch + self.stime + np.random.randint(self.tbatch)

        year = self.years[year_index]
        dfs_index = year_index * self.sbatch + sindex
        _, _, path = self.dfs[dfs_index]

        self.df = xr.open_dataset(path, engine="netcdf4")

        """self.logger.info(
                f"Torch batch index: {index}\n"
                f"  → Time block index: {tblock_index}\n"
                f"  → Selected time block (tblock): {tblock}\n"
                f"  → Year index: {year_index} → Year: {year}\n"
                f"  → Local tblock: {local_tblock}\n"
                f"  → Final time index (tindex): {tindex}\n"
                f"  → Spatial batch index (sindex): {sindex}\n"
                f"  → dfs index: {dfs_index}\n"
                f"  → File used: {path}"
                )

        """
        sequence_length_dim = self.min_dims['dim_2']
        dim_1 = self.min_dims['dim_1']
        dim_3 = self.min_dims['dim_3']
        dim_4 = self.min_dims['dim_4']
        self.schunk = dim_1 * dim_3 * dim_4

        npcosz = np.zeros([self.schunk, len(self.cosz), sequence_length_dim])
        nplai  = np.zeros([self.schunk, len(self.lai), sequence_length_dim])
        npssa  = np.zeros([self.schunk, len(self.ssa), sequence_length_dim])
        npov   = np.zeros([self.schunk, len(self.ov), sequence_length_dim])
        nprs   = np.zeros([self.schunk, len(self.rs), sequence_length_dim])


        for variable_index, variable_name in enumerate(self.cosz):
            da = self.df[variable_name]
            temp = da.isel(time=tindex, dim_1=slice(0, dim_1)).values
            temp = self.normalize(temp, variable_name)
            temp = np.tile(temp, dim_3 * dim_4)
            temp = np.tile(temp[:, np.newaxis], (1, sequence_length_dim))
            npcosz[:, variable_index, :] = temp
        tcosz = torch.tensor(npcosz, dtype=torch.float32)
        
        for variable_index, variable_name in enumerate(self.lai):
            da = self.df[variable_name]
            temp = da.isel(time=tindex, dim_1=slice(0, dim_1)).values
            temp = self.normalize(temp, variable_name)
            temp = temp.transpose(0, 2, 1)
            temp = temp.reshape(dim_3 * dim_1, sequence_length_dim)
            temp = np.tile(temp, (dim_4, 1))
            nplai[:, variable_index, :] = temp
        tlai = torch.tensor(nplai, dtype=torch.float32)
        
        for variable_index, variable_name in enumerate(self.ssa):
            da = self.df[variable_name]
            temp = da.isel(time=tindex).values
            temp = self.normalize(temp, variable_name)
            temp = temp.reshape(-1, 1)
            temp = np.tile(temp, (dim_1, 1))
            temp = np.tile(temp, (1, sequence_length_dim))
            npssa[:, variable_index, :] = temp    
        tssa = torch.tensor(npssa, dtype=torch.float32)
        
        for variable_index, variable_name in enumerate(self.rs):
            da = self.df[variable_name]
            temp = da.isel(time=tindex, dim_1=slice(0, dim_1)).values
            temp = self.normalize(temp, variable_name)
            temp = temp.reshape(-1, 1)
            temp = np.tile(temp, (1, sequence_length_dim))
            nprs[:, variable_index, :] = temp
        trs = torch.tensor(nprs, dtype=torch.float32)
        
        for variable_index, variable_name in enumerate(self.ov):
            da = self.df[variable_name]
            temp = da.isel(time=tindex, dim_1=slice(0, dim_1)).values
            temp = self.normalize(temp, variable_name)
            temp = temp.transpose(0, 2, 3, 1)
            temp = temp.reshape(-1, sequence_length_dim)
            npov[:, variable_index, :] = temp
        tov = torch.tensor(npov, dtype=torch.float32)
        
        feature = torch.cat([tcosz, tlai, tssa, trs],  dim=1)
        return (feature, tov)
