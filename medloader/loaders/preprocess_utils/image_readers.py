"""
Image reading tools and functions, facilitating loading of medical data
in different formats.
"""
import numpy as np
import dicom


def read_dicom_image(path: str):
    """
    Read dicom data in specific path.
    
    :param path: Path do dicom file to load.
    :returns: The loaded dicom file as a np array.
    """
    return dicom.dcmread(path).pixel_array.astype(np.float32)
