"""
Medical dataloader object.
"""
from typing import Optional
import os

import numpy as np

import torch
from torch.utils.data import Dataset


class MedicalDataloader(Dataset):
    """
    Class handler for Medical dataset loading, handling and processing.

    1. It is able to deal with datasets stored with specific folder structure for images and masks.
    2. Generates single slice or cubic slice dataset generators, via parameters 'cubic_slicer_context_layers'.
    3. Expects images data to be in dicom format (ATM - extendable to other formats).
    4. New datasets require the customization of the raw medical data loading and saving into .npy format,
    so that method 'read_dataset_pair_from_path' reads and preparares the output Dataset.
    """

    def __init__(
        self,
        logger,
        transform: bool,
        verbose: bool,
        dataset_path: str,
        folder_samples: list,
        scan_folder: str,
        mask_folder: str,
        cubic_slicer_context_layers: int,
        image_name_prefix: str,
        mask_name_prefix: str,
        target_shape: int,
        crop_shape: Optional[int] = None
    ):
        """
        """
        super(MedicalDataloader, self).__init__()
        self.logger = logger
        self.dataset_path = dataset_path
        self.folder_samples = folder_samples
        self.scan_folder = scan_folder
        self.mask_folder = mask_folder
        self.cubic_slicer_context_layers = cubic_slicer_context_layers
        self.image_name_prefix = image_name_prefix
        self.mask_name_prefix = mask_name_prefix
        self.transform = transform
        self.img_range_mean = None
        self.img_range_std = None
        self.mask_range_mean = None
        self.mask_range_std = None
        self.img_data_paths = dict()
        self.masks_data_paths = dict()
        self.verbose = verbose
        self.target_shape = target_shape
        self.inputs_shape = (self.cubic_slicer_context_layers*2 +1, self.target_shape, self.target_shape)  # C x H x W expected dimension
        self.crop_shape =  target_shape if crop_shape is None else crop_shape

        # Generate image data paths
        self._validate_runner_device()

        if self.verbose:  # Flag run to generate the final torch Dataset object with information.
            self.image_data_stats()

            self.scan_paths = list(self.img_data_paths.keys())
            #print("PATHS", self.img_data_paths)
            self.logger.debug(f"Medical dataset with {len(self.folder_samples)} patients. Raw shape {self.inputs_shape}. Output shape {self.crop_shape}.")

    def _validate_runner_device(self):
        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def __len__(self):
        """
        """
        return len(self.scan_paths)

    def read_cubic_data(self, idx: int):
        """
        """
        input_placeholder = np.zeros(self.inputs_shape)
        mask_placeholder = np.zeros(self.inputs_shape)

        for cubic_idx in range(len(self.img_data_paths[self.scan_paths[idx]])):
            idx_image, idx_mask = self.read_dataset_pair_from_path(
                image_path=self.img_data_paths[self.scan_paths[idx]][cubic_idx],
                label_path=self.masks_data_paths[self.scan_paths[idx]][cubic_idx]
            )

            input_placeholder[cubic_idx :, :] = idx_image[np.newaxis, :, :]
            mask_placeholder[cubic_idx, :, :] = idx_mask[np.newaxis, :, :]

        input_placeholder = input_placeholder[np.newaxis, :, :, :]
        mask_placeholder = mask_placeholder[np.newaxis, :, :, :]

        return {
            'image': input_placeholder.astype(np.int16),
            'target': mask_placeholder.astype(np.int16),
            'metadata': self.scan_paths[idx]
        }  # Return also the slice name of the corresponding input.

    def __getitem__(self, idx: int) -> dict:
        """
        """

        data = self.read_cubic_data(idx)
        return self.transform_processor(data)

    def read_dataset_pair_from_path(
            self,
            image_path,
            label_path
    ):
        """
        Read dicom or Nifty pairs of scan and corresponding mask, from paths.
        """

        if image_path.split('.')[-1] == "npy":
            image = np.load(
                image_path
            ).astype(np.int16)
            image = image[:self.target_shape, :self.target_shape]
            label = np.load(
                label_path
            ).astype(np.int16)
            label = label[:self.target_shape, :self.target_shape]

        return image, label

    def image_data_stats(self, data_type: str):
        """
        Method to gather image sample data paths into a dictionary variable
        self.img_data_paths and self.mask_data_paths. Also any necessary global statistics.
        """
        raise NotImplementedError

    def transform_processor(
            self,
            data: dict
    ):
        """
        Method to gather create processing transforms to apply.
        """
        raise NotImplementedError
