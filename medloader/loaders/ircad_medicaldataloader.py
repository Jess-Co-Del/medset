"""
Medical dataloader compatible with 3DIRcardb dataset.
"""

import os
import logging

import numpy as np
import pydicom as dicom

from monai import transforms

from .preprocess_utils.mask_edges import clip_data
from .preprocess_utils.image_readers import read_dicom_image

from .medicaldataloader import MedicalDataloader


class IrcadbDataloader(MedicalDataloader):
    """
    Medical Dataset orchestrator adapted to load the 3DIrcadB dataset
    with 20 CT scans available.
    """

    def dataset_clipping_preparation(self):
        """
        Preprocessing required for usage of the Lits Nifty files to be
        used by the MedicalDataLoader api. This function clips the images,
        to [300,300,:] resolution.
        """
        PATCH_SIZE = self.target_shape

        logging.basicConfig(
            level=logging.INFO,
            filename="/dataset/Lits/patching_log.log", filemode="w"
        )
        target_path = os.path.join(self.dataset_path, "clipped")
        os.makedirs(target_path, exist_ok=True)

        for scan_id in self.folder_samples:
            scan_output = os.path.join(target_path, f"{scan_id}")
            os.makedirs(scan_output, exist_ok=True)
            patient_folder, masks_folder = self.read_dicom_pair_paths(
                scan_id, raw_process=True)

            logging.info(f"Processing {patient_folder} with masks: {masks_folder}")

            for slice_id in range(len(os.listdir(patient_folder))):
                image, mask = self.read_dicom_pair(patient_folder, masks_folder, slice_id)

                logging.info(np.unique(mask, return_counts=True))

                image, mask = clip_data(image, mask, PATCH_SIZE)
                logging.info(np.unique(mask, return_counts=True))

                np.save(
                    os.path.join(scan_output, f"{self.image_name_prefix}{slice_id}.npy"),
                    image
                )
                np.save(
                    os.path.join(scan_output, f"{self.mask_name_prefix}{slice_id}.npy"),
                    mask
                )

    def read_dicom_pair(
        self,
        patient_folder: str,
        masks_folder: list,
        scan_id: str,
        ):
        """
        Reading dicoms for clippind dataset preparation.
        """
        patient_scan = read_dicom_image(
            os.path.join(patient_folder, f'{self.image_name_prefix}{scan_id}')
        )

        label_scan = np.zeros(patient_scan.shape)
        for folder in masks_folder:
            label_slice = read_dicom_image(
                os.path.join(folder, f'{self.image_name_prefix}{scan_id}')
            )
            if label_slice.max() == 255:
                label_slice /= 255
            if any(substring in folder for substring in [
                "tumor", "cyst", "metastasectomie", "kist"
            ]):
                label_slice *= 2

            label_scan += label_slice

            label_scan[label_scan > 2] = 2

        return patient_scan, label_scan

    def read_dicom_pair_paths(
        self,
        sample_id,
        raw_process: bool = True
        ):
        """
        
        """
        patient_folder = os.path.join(
                        self.dataset_path,
                        sample_id,
            )

        if raw_process:
            masks_folder = [
                os.path.join(
                    patient_folder, self.mask_folder, maskdir
                ) for maskdir in  os.listdir(
                    os.path.join(patient_folder, self.mask_folder)
                ) if any(substring in maskdir for substring in [
                    "liver", "cyst", "metastasectomie", "kist"
                ])
            ]

            patient_folder = os.path.join(
                        patient_folder,
                        self.scan_folder
            )

            return patient_folder, masks_folder
        else:
            return patient_folder

    def image_data_stats(self):
        """
        Method to gather sample data global statistics.
        Base class instantiates the following base statistics:
        self.img_range_mean = None
        self.img_range_std = None
        self.mask_range_mean = None
        self.mask_range_std = None
        """
        slices_list_values = dict()

        for sample_id in self.folder_samples:

            scan_folder = self.read_dicom_pair_paths(sample_id, raw_process=False)

            patient_slices = int(len(os.listdir(scan_folder))/2)
            slices_list_values[scan_folder] = patient_slices  # Log slice count

            # FIX ME LATER: se a primeira slice non-zero, fails.
            for slice_id in range(
                self.cubic_slicer_context_layers,
                patient_slices-self.cubic_slicer_context_layers
            ):
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

    def transform_processor(
        self,
        data: dict,
        ) -> dict:
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
                    transforms.ScaleIntensityRanged(
                        keys=["target"], a_min=0.0, a_max=2.0, b_min=0, b_max=2.0, clip=True
                    ),
                    transforms.RandSpatialCropd(  # CHECK ME LATER RandCropByPosNegLabeld()
                        keys=["image", "target"],
                        # label_key="targe"
                        roi_size=(-1, self.crop_shape, self.crop_shape),
                        random_size=False
                    ),
                    # transforms.RandAffined(
                    #     keys=["image", "target"],
                    #     mode=("bilinear", "nearest"),
                    #     prob=0.8,
                    #     shear_range=[[0,0],[0,0],[0.5,0.8]],
                    #     rotate_range=[[0,0],[0,0],[0,0]],
                    #     #scale_range=(0.15, 0.15, 0.15),
                    #     padding_mode="zeros"
                    # ),
                    transforms.RandRotateD(
                        keys=["image", "target"],
                        range_x=(-3.14, 3.14),
                        prob=0.9,
                        padding_mode='zeros'
                        #spatial_axes=(1,2)
                    ),
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
                transforms.RandSpatialCropd(
                    keys=["image", "target"],
                    roi_size=(-1, self.crop_shape, self.crop_shape),
                    random_size=False
                    # spatial_size=(-1, self.crop_shape, self.crop_shape),
                    # label_key="target",
                    # pos=1
                ),
                transforms.ToTensord(keys=["image", "target"]),
            ])

            transformed = base_transforms(data)

        return transformed
