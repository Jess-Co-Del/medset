import os
import pathlib

default_confs = {
    "note": "Attention UNET run with value trim",
    "generate_report_folder": True,
    "reporting_target_folder": "/etc/src/result_report",
    "datasets": {
        "cubic_slicer_context_layers": 1, # when =2, inputs include 5 slices. Two adjacent slices in the z-axis.
        "train": {
            "Lits": {
                "target_shape" : 300,
                "crop_shape" : 300,
                "dataset_path": "/dataset/Lits",
                "image_name_prefix": "volume_",
                "mask_name_prefix": "segmentation_",
                "folder_samples": [
                    # Overriden in code
                ],
                "scan_folder": "volume_",  # Replacement in string path, between scan_folder to mask_folder
                "mask_folder": "segmentation_",
            }
        },
        # Val configuration
        "val": {
            "ircadb": {
                "target_shape" : 300,
                "dataset_path": "/dataset/clipped",
                "image_name_prefix": "image_",
                "mask_name_prefix": "segmentation_",
                "folder_samples": ['3Dircadb1.5', '3Dircadb1.15','3Dircadb1.16','3Dircadb1.17',
                        '3Dircadb1.18','3Dircadb1.19', '3Dircadb1.20', '3Dircadb1.9', '3Dircadb1.2',
                        '3Dircadb1.3', '3Dircadb1.4', '3Dircadb1.6', '3Dircadb1.7', '3Dircadb1.8',
                        '3Dircadb1.10','3Dircadb1.11', '3Dircadb1.12', '3Dircadb1.13', '3Dircadb1.14'],
                "scan_folder": "PATIENT_DICOM",
                "mask_folder": "MASKS_DICOM",
            }
        },
    },
}

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Configuration(metaclass=Singleton):

    def __init__(self, configs: dict = None):
        if configs:
            self.data = configs
        else:
            self.data = {}

    def get(self, key: str) -> str:
        if key in self.data:
            return self.data[key]
        else:
            return default_confs[key]

    def set_config(self, key: str, value: str):
        self.data[key] = value
        return self

    def get_config(self):
        return self.data
