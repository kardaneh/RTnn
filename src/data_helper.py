import torch
from torch.utils.data import Dataset
import os
import numpy as np

class DataPreprocessor(Dataset):
    """
    """
    def __init__(
        self,
        logger,
        df,
        from_time,
        end_time,
        batch_divid_number,
        point_folds,
        time_folds,
        norm_mapping={},
        vertical_layers=57,
        point_number=10242,
        random_throw=False,
        only_layer=False,
    ):
        self.logger = logger
        self.df = df
        self.from_time = from_time
        self.end_time = end_time
        self.sbatch = batch_divid_number
        self.sfolds = point_folds
        self.tfolds = time_folds
        self.ngrid = point_number
        self.vrandom = random_throw
        self.only_layer = only_layer
        self.norm_mapping = norm_mapping
        self.vlevels = vertical_layers

        assert self.df.sizes['np'] == self.ngrid, f"Mismatch in spatial grid size: expected {self.ngrid}, got {self.df.sizes['np']}"
        assert self.df.sizes['nz1'] == self.vlevels, f"Mismatch in vertical levels: expected {self.vlevels}, got {self.df.sizes['nz1']}"

        self.logger.info(f"Time range: {self.from_time} ... {self.end_time}")
        self.logger.info(f"Spatial batches: {self.sbatch}")
        self.logger.info(f"Spatial folds: {self.sfolds}")
        self.logger.info(f"Temporal folds: {self.tfolds}")
        self.logger.info(f"Number of grid points: {self.ngrid}")
        self.logger.info(f"Random throw enabled: {self.vrandom}")
        self.logger.info(f"Only process one layer: {self.only_layer}")
        self.logger.info(f"Vertical levels: {self.vlevels}")


        # Spatial block
        self.sblock = self.ngrid // self.sbatch
        self.logger.info(f"Spatial block size: {self.sblock}")

        # Surface variables
        self.slv = [
            "aldif",
            "aldir",
            "asdif",
            "asdir",
            "cosz",
            "landfrac",
            "sicefrac",
            "snow",
            "solc",
            "tsfc",
            "emiss",
        ]

        # Spatial chunk
        self.schunk = self.sblock // self.sfolds
        self.logger.info(f"Spatial chunk size: {self.schunk}")

        # single-level array
        self.npslv = np.zeros([self.schunk, len(self.slv), 1])

        # Multi-level variables
        self.mlv = [
            "ccl4vmr",
            "cfc11vmr",
            "cfc12vmr",
            "cfc22vmr",
            "ch4vmr",
            "cldfrac",
            "co2vmr",
            "n2ovmr",
            "o2vmr",
            "o3vmr",
            "play",
            "qc",
            "qg",
            "qi",
            "qr",
            "qs",
            "qv",
            "tlay",
        ]

        # Multi-level cumulative variables
        self.mlcv = {"cldfrac": 0, "qc": 1}

        # Output variable
        self.ov = ["swuflx", "swdflx", "lwuflx", "lwdflx"]

        # Auxiliary variables (pressure / absoption levels)
        self.auxv = ["plev"]

    def __len__(self):
        """Returns the number of """
        return ((self.end_time - self.from_time)// self.tfolds * self.sbatch)

    def __getitem__(self, index):
        """
        """
        #self.logger.info(f"Torch batch: {index}\n")

        # Random vertically ? 
        if self.vrandom == True:
            start, last, middle= 0, self.vlevels -1, self.vlevels - 2
            rvlevslen = middle - np.random.choice(middle//2)
            rvlevs = np.random.choice(np.arange(start+1, last), size=rvlevslen, replace=False)
            rvlevs.sort()
            vlevs = np.concatenate([np.asarray([start]), rvlevs, np.asarray([last])])
        else:
            vlevs = np.arange(0, 57)

        vleys = vlevs[0:-1]

        # Initialize
        self.npmlv  = np.zeros([self.schunk,     len(self.mlv ), len(vlevs)])
        self.npmlcv = np.zeros([self.schunk, 2 * len(self.mlcv), len(vlevs)])
        self.npov   = np.zeros([self.schunk,     len(self.ov  ), len(vlevs)])
        self.npauxv = np.zeros([self.schunk,     len(self.auxv), len(vlevs)])

        time_index = self.from_time + np.random.randint(self.tfolds)
        gindices = np.arange(index * self.sblock, index * self.sblock + self.sblock)
        assert len(gindices) == self.sblock, f"Mismatch in number of spatial blocks: expected {self.sblock}, got {len(gindices)}"

        # Random shift in spatial batch
        sindex = np.random.randint(self.sfolds)
        bindices = np.arange(sindex, sindex + self.schunk * self.sfolds, self.sfolds)
        assert len(bindices) == self.schunk, f"Mismatch in number of batch chunks: expected {self.schunk}, got {len(bindices)}"

        # Single level variables
        for variable_index, variable_name in enumerate(self.slv):
            if variable_name == "emiss":
                temp = (
                        self.df.variables[variable_name][time_index, gindices, 0]
                        - self.norm_mapping[variable_name]["mean"]
                        ) / self.norm_mapping[variable_name]["std"]
                self.npslv[:, variable_index, 0] = temp[bindices]
            else:
                temp = (
                        self.df.variables[variable_name][time_index, gindices]
                        - self.norm_mapping[variable_name]["mean"]
                        ) / self.norm_mapping[variable_name]["std"]
                self.npslv[:, variable_index, 0] = temp[bindices]
        tslv = torch.tensor(self.npslv, dtype=torch.float32)

        # Multi level variables
        for variable_index, variable_name in enumerate(self.mlv):
            temp = (
                    self.df.variables[variable_name][time_index, gindices, :]
                    - self.norm_mapping[variable_name]["mean"]
                    ) / self.norm_mapping[variable_name]["std"]
            temp_value = temp[bindices][:, vleys]
            self.npmlv[:, variable_index, 1 : len(vlevs)] = temp_value
            self.npmlv[:, variable_index, 0] = self.npmlv[:, variable_index, 1]

            if variable_name in self.mlcv:
                variable_index = self.mlcv[variable_name]
                temp_value_cumsum_forward = np.cumsum(temp_value, axis=1) / 20.0
                temp_value_cumsum_backward = np.cumsum(temp_value[:, ::-1], axis=1) / 20.0
                self.npmlcv[:, variable_index, 1 : len(vlevs)] = temp_value_cumsum_forward
                self.npmlcv[:, variable_index, 0] = self.npmlcv[:, variable_index, 1]
                self.npmlcv[:, len(self.mlcv) + variable_index, 1 : len(vlevs)] = temp_value_cumsum_backward
                self.npmlcv[:, len(self.mlcv) + variable_index, 0] = self.npmlcv[:, variable_index, 1]

        tmlv = torch.tensor(self.npmlv, dtype=torch.float32)
        tmlcv = torch.tensor(self.npmlcv, dtype=torch.float32)

        # Output variables
        for variable_index, variable_name in enumerate(self.ov):
            temp = (
                    self.df.variables[variable_name][time_index, gindices, :]
                    - self.norm_mapping[variable_name]["mean"]
                    ) / self.norm_mapping[variable_name]["std"]
            self.npov[:, variable_index, :] = temp[bindices][:, vlevs]
        tov = torch.tensor(self.npov, dtype=torch.float32)

        # Auxiliary variables
        for variable_index, variable_name in enumerate(self.auxv):
            temp = self.df.variables[variable_name][time_index, gindices, :]
            self.npauxv[:, variable_index, :] = temp[bindices][:, vlevs]
        tauxv = torch.tensor(self.npauxv, dtype=torch.float32)

        # Pressure difference
        p_diff = tauxv - torch.roll(tauxv, -1, 2)
        p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        # Feature combination
        if self.only_layer:
            feature = torch.cat([torch.tile(tslv, [1, len(vlevs)]), tmlv], dim=1)
        else:
            feature = torch.cat([torch.tile(tslv, [1, len(vlevs)]), tmlv, (p_diff - 17.2) / 9.8, tmlcv], dim=1)

        return (feature, tov, tauxv)
