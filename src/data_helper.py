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
        """
        1  aldif Surface albedo (near-infrared spectral regions) for diffuse radiation
        2  aldir Surface albedo (near-infrared spectral regions) for direct radiation
        3  asdif Surface albedo (UV/visible spectral regions) for diffuse radiation
        4  asdir Surface albedo (UV/visible spectral regions) for direct radiation
        5  cosz Cosine solar zenith angle for current time step
        6  landfrac Land mask (1 for land, 0 for water)
        7  sicefrac Sea ice fraction
        8  snow Snow water depth
        9  solc Solar constant
        10 tsfc Surface temperature
        11 emiss Surface emissivity for 16 LW spectral bands
        """
        self.single_height_variable = [
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
        self.points_in_batch = self.block_size // self.point_folds - 1

        # Initialize single-feature array
        self.single_feature = np.zeros(
            [self.points_in_batch, len(self.single_height_variable), 1]
        )

        # Multi-height variables
        """
        1  ccl4vmr CCL4 volume mixing ratio Layer
        2  cfc11vmr CFC11 volume mixing ratio Layer
        3  cfc12vmr CFC12 volume mixing ratio Layer
        4  cfc22vmr CFC22 volume mixing ratio Layer
        5  ch4vmr Methane volume mixing ratio Layer
        6  cldfrac Cloud fraction Layer
        7  CO2vmr CO2 volume mixing ratio Layer
        8  N2Ovmr N2O volume mixing ratio Layer
        9  O2vmr O2 volume mixing ratio Layer
        10 o3vmr O3 volume mixing ratio Layer
        11 play Layer pressure Layer hPa
        12 tlay Layer temperature Layer
        13 qc Cloud water mixing ratio Layer
        14 qg Graupel mixing ratio Layer
        15 qi Cloud ice mixing ratio Layer
        16 qr Rain water mixing ratio Layer
        17 qs Snow mixing ratio Layer
        18 qv Water vapor mixing ratio Layer
        """
        self.multi_height_variable = [
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
        self.multi_height_cumsum_variable = {"cldfrac": 0, "qc": 1}

        # Multi-height cumulative sum variables
        """
        1 swuflx Layer SW upward fluxes
        2 swdflx Layer SW downward fluxes
        3 lwuflx Layer LW upward fluxes
        4 lwdflx Layer LW downward fluxes
        """
        self.label_variable = ["swuflx", "swdflx", "lwuflx", "lwdflx"]

        # Auxiliary variables (pressure levels)
        self.auxiliary_variable = ["plev"]

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
            keep_size = 55 - np.random.choice(25)
            keep_index = np.concatenate(
                [
                    np.asarray([0]),
                    np.random.choice(np.arange(1, 56), size=keep_size, replace=False),
                    np.asarray([56]),
                ]
            )
            keep_index.sort()
            keep_index_level = keep_index
            keep_index_layer = keep_index_level[0:-1]

        else:
            keep_index_level = np.arange(0, 57)
            keep_index_layer = np.arange(0, 56)

        # Initialize feature arrays
        self.multi_feature = np.zeros(
            [
                self.points_in_batch,
                len(self.multi_height_variable),
                len(keep_index_level),
            ]
        )
        self.multi_cumsum_feature = np.zeros(
            [
                self.points_in_batch,
                2 * len(self.multi_height_cumsum_variable),
                len(keep_index_level),
            ]
        )
        self.label_feature = np.zeros(
            [self.points_in_batch, len(self.label_variable), len(keep_index_level)]
        )
        self.auxiliary_feature = np.zeros(
            [self.points_in_batch, len(self.auxiliary_variable), len(keep_index_level)]
        )

        time_index = (
            (index // self.batch_divid_number) * self.time_folds
            + self.from_time
            + np.random.randint(self.time_folds)
        )
        remain_index = index % self.batch_divid_number
        total_folds = self.block_size // self.point_folds
        global_index_list = np.arange(
            (remain_index * self.block_size),
            (remain_index * self.block_size + self.block_size),
        )

        # Random index shift for batching
        inside_start_shift = np.random.randint(self.point_folds)
        inside_index_list = np.arange(
            inside_start_shift,
            inside_start_shift + self.points_in_batch * self.point_folds,
            self.point_folds,
        )
        index_list = np.arange(
            (remain_index * self.block_size),
            (remain_index * self.block_size + total_folds * self.point_folds),
            self.point_folds,
        )

        # Single feature processing
        for variable_index, variable_name in enumerate(self.single_height_variable):
            if variable_name == "emiss":
                temp = (
                    df.variables[variable_name][time_index, global_index_list, 0]
                    - self.norm_mapping[variable_name]["mean"]
                ) / self.norm_mapping[variable_name]["scale"]
                self.single_feature[:, variable_index, 0] = temp[inside_index_list]
            else:
                temp = (
                    df.variables[variable_name][time_index, global_index_list]
                    - self.norm_mapping[variable_name]["mean"]
                ) / self.norm_mapping[variable_name]["scale"]
                self.single_feature[:, variable_index, 0] = temp[inside_index_list]

        single_feature_tt = torch.tensor(self.single_feature, dtype=torch.float32)

        # Multi-feature processing
        for variable_index, variable_name in enumerate(self.multi_height_variable):
            temp = (
                df.variables[variable_name][time_index, global_index_list, :]
                - self.norm_mapping[variable_name]["mean"]
            ) / self.norm_mapping[variable_name]["scale"]
            temp_value = np.array(temp[inside_index_list, ::]).take(
                keep_index_layer, axis=1
            )
            self.multi_feature[:, variable_index, 1 : len(keep_index_layer) + 1] = (
                temp_value
            )
            self.multi_feature[:, variable_index, 0] = self.multi_feature[
                :, variable_index, 1
            ]

            if variable_name in self.multi_height_cumsum_variable:
                variable_index = self.multi_height_cumsum_variable[variable_name]
                temp_value_cumsum_forward = np.cumsum(temp_value, axis=1) / 20.0
                temp_value_cumsum_backward = (
                    np.cumsum(temp_value[:, ::-1], axis=1) / 20.0
                )
                self.multi_cumsum_feature[
                    :, variable_index, 1 : len(keep_index_level)
                ] = temp_value_cumsum_forward
                self.multi_cumsum_feature[:, variable_index, 0] = (
                    self.multi_cumsum_feature[:, variable_index, 1]
                )
                self.multi_cumsum_feature[
                    :,
                    len(self.multi_height_cumsum_variable) + variable_index,
                    1 : len(keep_index_level),
                ] = temp_value_cumsum_backward
                self.multi_cumsum_feature[
                    :, len(self.multi_height_cumsum_variable) + variable_index, 0
                ] = self.multi_cumsum_feature[:, variable_index, 1]

        multi_feature_tt = torch.tensor(self.multi_feature, dtype=torch.float32)
        multi_cumsum_feature_tt = torch.tensor(
            self.multi_cumsum_feature, dtype=torch.float32
        )

        # Label feature processing
        for variable_index, variable_name in enumerate(self.label_variable):
            temp = (
                df.variables[variable_name][time_index, global_index_list, :]
                - self.norm_mapping[variable_name]["mean"]
            ) / self.norm_mapping[variable_name]["scale"]
            self.label_feature[:, variable_index, :] = np.array(
                temp[inside_index_list, ::]
            ).take(keep_index_level, axis=1)

        label_feature_tt = torch.tensor(self.label_feature, dtype=torch.float32)

        # Auxiliary feature processing
        for variable_index, variable_name in enumerate(self.auxiliary_variable):
            temp = df.variables[variable_name][time_index, global_index_list, :]
            self.auxiliary_feature[:, variable_index, :] = np.array(
                temp[inside_index_list, ::]
            ).take(keep_index_level, axis=1)

        auxiliary_result_tf = torch.tensor(self.auxiliary_feature, dtype=torch.float32)

        # Compute pressure difference
        p_diff = auxiliary_result_tf - torch.roll(auxiliary_result_tf, -1, 2)
        p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        # Feature combination based on layer condition
        if self.only_layer:
            feature_tf = torch.cat(
                [
                    torch.tile(single_feature_tt, [1, len(keep_index_level)]),
                    multi_feature_tt,
                ],
                dim=(1),
            )
        else:
            feature_tf = torch.cat(
                [
                    torch.tile(single_feature_tt, [1, len(keep_index_level)]),
                    multi_feature_tt,
                    (p_diff - 17.2) / 9.8,
                    multi_cumsum_feature_tt,
                ],
                dim=(1),
            )

        return (feature_tf, label_feature_tt, auxiliary_result_tf)
