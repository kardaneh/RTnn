import torch
from torch.utils.data import Dataset
import os
import numpy as np
import xarray as xr


class RtmMpasDatasetWholeTimeLarge(Dataset):
    """
    A dataset class for handling large-scale data processing in time-series format.

    Arguments:
        nc_file (str): NetCDF file path.
        root_dir (str): Root directory for the dataset.
        from_time (int): Starting time index.
        end_time (int): Ending time index.
        batch_divid_number (int): The number of divisions for batching.
        point_folds (int): The size of the spatial blocks.
        time_folds (int): The size of the time blocks.
        norm_mapping (dict): Normalization parameters (mean and scale).
        vertical_layers (int): Number of vertical layers (default 57).
        point_number (int): Total number of points (default 10242).
        random_throw (bool): Whether to apply random throwing (default False).
        only_layer (bool): Whether to return only layer features (default False).
    """

    def __init__(
        self,
        nc_file,
        root_dir,
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
        self.nc_file = nc_file
        self.root_dir = root_dir
        self.from_time = from_time
        self.end_time = end_time
        self.batch_divid_number = batch_divid_number
        self.point_folds = point_folds
        self.time_folds = time_folds
        self.point_number = point_number
        self.random_throw = random_throw
        self.only_layer = only_layer
        self.norm_mapping = norm_mapping
        self.vertical_layers = vertical_layers
        print(f"Vertical layers: {self.vertical_layers}")

        self.block_size = self.point_number // self.batch_divid_number

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

        # Points in batch
        self.gpbb = self.block_size // self.point_folds - 1

        # Initialize single-feature array
        self.sf = np.zeros([self.gpbb, len(self.slv), 1])

        # Multi-height variables
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

        # Multi-height cumulative sum variables
        self.mlcv = {"cldfrac": 0, "qc": 1}

        # Output variable
        self.ov = ["swuflx", "swdflx", "lwuflx", "lwdflx"]

        # Auxiliary variables (pressure levels)
        self.auxv = ["plev"]

    def __len__(self):
        """Returns the number of batches in the dataset."""
        return (
            (self.end_time - self.from_time)
            // self.time_folds
            * self.batch_divid_number
        )

    def __getitem__(self, index):
        """Fetches the data for a specific batch.
        index: index in <some list of batch_size indices from 0 to __len__> passed by PyTorch
        """
        full_file_path = os.path.join(self.root_dir, self.nc_file)
        df = xr.open_dataset(full_file_path, engine="netcdf4")

        # Random throwing for feature selection
        if self.random_throw == True:
            # np.random.choice(25) gives a random integer from 0 to 24, 
            # rlen becomes a random number from 31 (i.e., 55-24) to 55 (i.e., 55-0)
            rlen = self.vertical_layers - 2 - np.random.choice(25)
            rinx = np.concatenate([np.asarray([0]), np.random.choice(np.arange(1, 56), size=rlen, replace=False), np.asarray([56])])
            rinx.sort()
            rinx_lev = rinx
            rinx_ley = rinx_lev[0:-1]

        else:
            rinx_lev = np.arange(0, 57)

        rinx_ley = rinx_lev[0:-1]

        # Initialize feature arrays
        self.npmlv = np.zeros([self.gpbb, len(self.mlv), len(rinx_lev)])
        self.npmlcv = np.zeros([self.gpbb, 2 * len(self.mlcv), len(rinx_lev)])
        self.npov = np.zeros([self.gpbb, len(self.ov), len(rinx_lev)])
        self.npauxv = np.zeros([self.gpbb, len(self.auxv), len(rinx_lev)])

        tinx = self.from_time + np.random.randint(self.time_folds)
        total_folds = self.block_size // self.point_folds
        ginx_lst = np.arange(index * self.block_size, index * self.block_size + self.block_size)

        # Random index shift for batching
        srinx = np.random.randint(self.point_folds)
        rinx_lst = np.arange(srinx, srinx + self.gpbb * self.point_folds, self.point_folds)

        # Single feature processing
        for variable_index, variable_name in enumerate(self.slv):
            if variable_name == "emiss":
                temp = (
                    df.variables[variable_name][tinx, ginx_lst, 0]
                    - self.norm_mapping[variable_name]["mean"]
                ) / self.norm_mapping[variable_name]["scale"]
                self.sf[:, variable_index, 0] = temp[rinx_lst]
            else:
                temp = (
                    df.variables[variable_name][tinx, ginx_lst]
                    - self.norm_mapping[variable_name]["mean"]
                ) / self.norm_mapping[variable_name]["scale"]
                self.sf[:, variable_index, 0] = temp[rinx_lst]

        sf_tt = torch.tensor(self.sf, dtype=torch.float32)

        # Multi-feature processing
        for variable_index, variable_name in enumerate(self.mlv):
            temp = (
                df.variables[variable_name][tinx, ginx_lst, :]
                - self.norm_mapping[variable_name]["mean"]
            ) / self.norm_mapping[variable_name]["scale"]
            temp_value = np.array(temp[rinx_lst, ::]).take(
                rinx_ley, axis=1
            )
            self.npmlv[:, variable_index, 1 : len(rinx_ley) + 1] = (
                temp_value
            )
            self.npmlv[:, variable_index, 0] = self.npmlv[
                :, variable_index, 1
            ]

            if variable_name in self.mlcv:
                variable_index = self.mlcv[variable_name]
                temp_value_cumsum_forward = np.cumsum(temp_value, axis=1) / 20.0
                temp_value_cumsum_backward = (
                    np.cumsum(temp_value[:, ::-1], axis=1) / 20.0
                )
                self.npmlcv[
                    :, variable_index, 1 : len(rinx_lev)
                ] = temp_value_cumsum_forward
                self.npmlcv[:, variable_index, 0] = (
                    self.npmlcv[:, variable_index, 1]
                )
                self.npmlcv[
                    :,
                    len(self.mlcv) + variable_index,
                    1 : len(rinx_lev),
                ] = temp_value_cumsum_backward
                self.npmlcv[
                    :, len(self.mlcv) + variable_index, 0
                ] = self.npmlcv[:, variable_index, 1]

        mf_tt = torch.tensor(self.npmlv, dtype=torch.float32)
        npmlcv_tt = torch.tensor(
            self.npmlcv, dtype=torch.float32
        )

        # Label feature processing
        for variable_index, variable_name in enumerate(self.ov):
            temp = (
                df.variables[variable_name][tinx, ginx_lst, :]
                - self.norm_mapping[variable_name]["mean"]
            ) / self.norm_mapping[variable_name]["scale"]
            self.npov[:, variable_index, :] = np.array(
                temp[rinx_lst, ::]
            ).take(rinx_lev, axis=1)

        npov_tt = torch.tensor(self.npov, dtype=torch.float32)

        # Auxiliary feature processing
        for variable_index, variable_name in enumerate(self.auxv):
            temp = df.variables[variable_name][time_index, ginx_lst, :]
            self.npauxv[:, variable_index, :] = np.array(
                temp[rinx_lst, ::]
            ).take(rinx_lev, axis=1)

        auxiliary_result_tf = torch.tensor(self.npauxv, dtype=torch.float32)

        # Compute pressure difference
        p_diff = auxiliary_result_tf - torch.roll(auxiliary_result_tf, -1, 2)
        p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        # Feature combination based on layer condition
        if self.only_layer:
            feature_tf = torch.cat(
                [
                    torch.tile(sf_tt, [1, len(rinx_lev)]),
                    mf_tt,
                ],
                dim=(1),
            )
        else:
            feature_tf = torch.cat(
                [
                    torch.tile(sf_tt, [1, len(rinx_lev)]),
                    mf_tt,
                    (p_diff - 17.2) / 9.8,
                    npmlcv_tt,
                ],
                dim=(1),
            )

        return (feature_tf, npov_tt, auxiliary_result_tf)
