"""
Medical dataloader compatible with Lits dataset.
"""

import os
import logging
import numpy as np
import math

from monai import transforms
from .medicaldataloader import MedicalDataloader
from .preprocess_utils.mask_edges import clip_data
import nibabel as nib


class LitsDataloader(MedicalDataloader):
    """
    Medical Dataset orchestrator adapted to load the Lits dataset
    with 131 CT scans available.
    
    Each Lits file come in a .nii file containing all slices of each scan.
    These are broken down into individual images in 'scans_to_image_files()'.
    These are then used normaly by the MedicalDataLoader object.
    """

    def dataset_clipping(self):
        """
        Preprocessing required for usage of the Lits Nifty files to be
        used by the MedicalDataLoader api. This function clips the images.
        Expects volume slices in dim=-1.
        """
        PATCH_SIZE = self.target_shape

        logging.basicConfig(level=logging.INFO, filename="/dataset/Lits/patching_log.log", filemode="w")
        target_path= os.path.join(self.dataset_path, f"clipped_{self.target_shape}_")
        os.makedirs(target_path, exist_ok=True)

        raw_folder = os.path.join(
            self.dataset_path,
            "raw"
        )

        for scan_id in range(0, 130):
            scan_output = os.path.join(target_path, f"scan_{scan_id}")
            os.makedirs(scan_output, exist_ok=True)
            mask = nib.load(os.path.join(raw_folder, f"segmentation-{scan_id}"+".nii")).get_fdata()
            image = nib.load(os.path.join(raw_folder, f"volume-{scan_id}"+".nii")).get_fdata()
            logging.info(f"Processing {os.path.join(raw_folder, f'volume-{scan_id}.nii')} with shape {image.shape}")

            image, mask = clip_data(image, mask, PATCH_SIZE)
            logging.info(f"Saving {os.path.join(raw_folder, f'volume-{scan_id}.nii')} with shape {image.shape}")
            for slice_id in range(image.shape[-1]):
                np.save(
                    os.path.join(scan_output, f"{self.image_name_prefix}{slice_id}.npy"),
                    image[:,:,slice_id]
                )
                np.save(
                    os.path.join(scan_output, f"{self.mask_name_prefix}{slice_id}.npy"),
                    mask[:,:,slice_id]
                )

    def dataset_patching_preparation(self):
        """
        This function patches the images, to [96,96,:] resolution. It analyses only LESION masks.
        Preprocessing required for usage of the Lits Nifty files to be
        used by the MedicalDataLoader api.
        """
        PATCH_SIZE = 96

        logging.basicConfig(level=logging.INFO, filename="/dataset/Lits/patching_log.log", filemode="w")
        target_path= os.path.join(self.dataset_path, "patched")
        os.makedirs(target_path, exist_ok=True)

        raw_folder = os.path.join(
            self.dataset_path,
            "raw"
        )

        for scan_id in range(1,130):
            scan_output = os.path.join(target_path, f"scan_{scan_id}")
            os.makedirs(scan_output, exist_ok=True)
            for f_string_file in [f"segmentation-{scan_id}", f"volume-{scan_id}"]:
                scan = nib.load(os.path.join(raw_folder, f_string_file+".nii")).get_fdata()
                logging.debug(f"Processing scan_{scan_id} with shape {scan.shape}")

                # Look at lesion labels ONLY.
                scan = scan - 1
                scan[scan < 0] = 0

                if "segmentation-" in f_string_file:  # Process masks first to get masks.
                    dims = np.nonzero(scan)
                    start_mask = [min(dim) for dim in dims]
                    end_mask = [max(dim) for dim in dims]
                    ranges = [end_mask[idx]-start_mask[idx] for idx in range(len(start_mask))]
                    start_centered = [start_mask[i] -math.floor((PATCH_SIZE - ranges[i])/2) for i in  range(len(start_mask))]
                    end_centered = [end_mask[i] + math.ceil((PATCH_SIZE - ranges[i])/2) for i in  range(len(start_mask))]

                for slice_id in range(scan.shape[2]):
                    np.save(
                        os.path.join(scan_output, f_string_file.split('-')[0]+f"_{slice_id}.npy"),
                        scan[
                            start_centered[0]:end_centered[0],
                            start_centered[1]:end_centered[1],
                            slice_id
                        ]
                    )

    def scans_to_image_files(self):
        """
        Preprocessing required for usage of the Lits Nifty files to be
        used by the MedicalDataLoader api.
        """
        logging.basicConfig(level=logging.INFO, filename="/dataset/Lits/nii_read_log.log",filemode="w")
        target_path = os.path.join(self.dataset_path, "processed")
        os.makedirs(target_path, exist_ok=True)

        raw_folder = os.path.join(
            self.dataset_path,
            "raw"
        )

        for scan_id in range(1,130):
            scan_output = os.path.join(target_path, f"scan_{scan_id}")
            os.makedirs(scan_output, exist_ok=True)
            for f_string_file in [f"volume-{scan_id}", f"segmentation-{scan_id}"]:
                scan = nib.load(os.path.join(raw_folder, f_string_file + ".nii")).get_fdata()
                logging.debug(f"Processing scan_{scan_id} with shape {scan.shape}")

                for slice_id in range(scan.shape[2]):
                    np.save(
                        os.path.join(scan_output, f_string_file.split('-')[0]+f"_{slice_id}.npy"),
                        scan[:,:,slice_id]
                    )

    def image_data_stats(self):
        """
        Method to gather sample data global statistics.
        Base class instantiates the following base statistics:
        self.img_range_mean = None
        self.img_range_std = None
        self.mask_range_mean = None
        self.mask_range_std = None

        :param data_type: String flag to distinguish the metric logic to apply to "images" and "masks".
        """
        slices_list_values = dict()

        dataset_path = self.dataset_path + '/clipped'
        self.folder_samples = os.listdir(dataset_path)
        for sample_id in self.folder_samples:

            scan_folder = os.path.join(
                dataset_path,
                sample_id
            )

            patient_slices = int(len(os.listdir(scan_folder))/2)
            slices_list_values[scan_folder] = patient_slices  # Log slice count

            for slice_id in range(patient_slices):
                # 1. Images of patient
                label_scan = np.load(
                    os.path.join(
                            scan_folder,
                            f'{self.mask_name_prefix}{slice_id}.npy')
                ).astype(float)

                if label_scan.sum() > 0:  # ONLY LOOK AT IMAGES WHERE THERE ARE MASKS
                    self.img_data_paths[os.path.join(
                        scan_folder,
                        f'{self.image_name_prefix}{slice_id}.npy'
                    )] = tuple(
                        os.path.join(
                            scan_folder,
                            f'{self.image_name_prefix}{context_id}.npy'
                        ) for context_id in range(
                            slice_id-self.cubic_slicer_context_layers,
                            slice_id+self.cubic_slicer_context_layers+1,
                        )
                    )

                    # 2. Masks of patient
                    self.masks_data_paths[os.path.join(
                        scan_folder,
                        f'{self.image_name_prefix}{slice_id}.npy'
                    )] = tuple(
                        os.path.join(
                            scan_folder,
                            f'{self.mask_name_prefix}{context_id}.npy'
                        ) for context_id in range(
                            slice_id-self.cubic_slicer_context_layers,
                            slice_id+self.cubic_slicer_context_layers+1,
                        )
                    )

        self.logger.debug(f"Dataset shape {slices_list_values}.")

    def transform_processor(self, data: dict) -> dict:
        """_summary_

        Args:
            data (_type_): _description_
        """

        if self.transform:

            train_transforms = transforms.Compose(
                [
                    transforms.ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
                    ),
                    # transforms.ScaleIntensityRanged(
                    #     keys=["target"], a_min=0.0, a_max=2.0, b_min=0, b_max=2.0, clip=True
                    # ),
                    # transforms.RandSpatialCropd(  # CHECK ME LATER RandCropByPosNegLabeld()
                    #     keys=["image", "target"],
                    #     # label_key="targe"
                    #     roi_size=(-1, self.crop_shape, self.crop_shape),
                    #     random_size=False
                    # ),
                    # transforms.RandAffined(
                    #     keys=["image", "target"],
                    #     mode=("bilinear", "nearest"),
                    #     prob=0.8,
                    #     shear_range=[[0,0],[0,0],[0.5,0.8]],
                    #     rotate_range=[[0,0],[0,0],[0,0]],
                    #     #scale_range=(0.15, 0.15, 0.15),
                    #     padding_mode="zeros"
                    # ),
                    # TRANSFORMS.RANDROTATED(
                    #     KEYS=["IMAGE", "TARGET"],
                    #     RANGE_X=(-3.14, 3.14),
                    #     PROB=0.9,
                    #     PADDING_MODE='ZEROS'
                    #     #SPATIAL_AXES=(1,2)
                    # ),
                    transforms.ToTensord(keys=["image", "target"]),
                ]
            )
            transformed = train_transforms(data)

        else:
            base_transforms = transforms.Compose([
                transforms.ScaleIntensityRanged(
                        keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
                    ),
                transforms.ScaleIntensityRanged(
                    keys=["target"], a_min=0.0, a_max=2.0, b_min=0, b_max=2.0, clip=True
                ),
                # transforms.RandSpatialCropd(
                #     keys=["image", "target"],
                #     roi_size=(-1, self.crop_shape, self.crop_shape),
                #     random_size=False
                #     # spatial_size=(-1, self.crop_shape, self.crop_shape),
                #     # label_key="target",
                #     # pos=1
                # ),
                transforms.ToTensord(keys=["image", "target"]),
            ])

            transformed = base_transforms(data)

        return transformed
