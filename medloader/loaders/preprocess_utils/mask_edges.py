# Derive GT of boundary (thin - 2*2)

import os, math
import cv2
import torch
from PIL import Image
import numpy as np
from skimage.morphology import remove_small_objects


def blob_cleaner(
    binary_mask: torch.Tensor,
    min_size: int = 64,
    connectivity: int = 1
):
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.int().cpu().numpy()

    # 1 - Clean by object detection
    binary_mask = remove_small_objects(
        binary_mask.astype(bool),
        min_size=min_size,
        connectivity=connectivity
    )
    binary_mask = (binary_mask*1).astype('uint8')

    # 2 - Clean by Erosion ncv2.erode() method
    # kernel = np.ones((2, 2), np.uint8)
    # binary_mask = cv2.erode(binary_mask, kernel)

    binary_mask = torch.from_numpy(binary_mask)
    return binary_mask


def extract_bboxes(mask):
    """
    Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

    for i in range(mask.shape[-1]):

        m = mask[:, :, i]

        # Bounding box.

        horizontal_indicies = np.where(np.any(m, axis=0))[0]

        #print("np.any(m, axis=0)",np.any(m, axis=0))

        #print("p.where(np.any(m, axis=0))",np.where(np.any(m, axis=0)))

        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:

            x1, x2 = horizontal_indicies[[0, -1]]

            y1, y2 = vertical_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1.

            x2 += 1

            y2 += 1

        else:

            # No mask for this instance. Might happen due to

            # resizing or cropping. Set bbox to zeros

            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def clip_data(volume, segmentation, PATCH_SIZE):
    """
    Clip images and masks, centering output in the masks object.
    Assumes slices are in dim=-1.
    """
    # Center of volume
    CENTER_PATCHING = slice(
        PATCH_SIZE,
        -PATCH_SIZE
    )

    if segmentation.sum() > 0:
        volume = np.pad(volume, PATCH_SIZE, mode='minimum')
        segmentation = np.pad(segmentation, PATCH_SIZE, mode='minimum')

        # BBOX calculus
        z = segmentation.sum(axis=(-3,-2)).argmax()
        mask = segmentation[:, :, z]

        bbox = extract_bboxes(mask[:, :, np.newaxis])
        # new bbox
        x_pad = (PATCH_SIZE - (bbox[0][2]-bbox[0][0]))/2
        y_pad = (PATCH_SIZE - (bbox[0][3]-bbox[0][1]))/2

        clipped_bbox = (
            math.ceil(bbox[0][0] - x_pad),
            math.ceil(bbox[0][2] + x_pad),
            math.ceil(bbox[0][1] - y_pad),
            math.ceil(bbox[0][3] + y_pad)
        )

        clipped_mask = segmentation[
            clipped_bbox[0]:clipped_bbox[1],
            clipped_bbox[2]:clipped_bbox[3],
            CENTER_PATCHING
        ]
        clipped_image = volume[
            clipped_bbox[0]:clipped_bbox[1],
            clipped_bbox[2]:clipped_bbox[3],
            CENTER_PATCHING
        ]
    else:
        HALF_PATCH_SIZE = PATCH_SIZE/2

        clipped_mask = segmentation[
            segmentation.shape[0]/2-HALF_PATCH_SIZE:segmentation.shape[0]/2+HALF_PATCH_SIZE,
            segmentation.shape[1]/2-HALF_PATCH_SIZE:segmentation.shape[1]/2+HALF_PATCH_SIZE,
            segmentation.shape[2]/2-HALF_PATCH_SIZE:segmentation.shape[2]/2+HALF_PATCH_SIZE
        ]
        clipped_image = volume[
            segmentation.shape[0]/2-HALF_PATCH_SIZE:segmentation.shape[0]/2+HALF_PATCH_SIZE,
            segmentation.shape[1]/2-HALF_PATCH_SIZE:segmentation.shape[1]/2+HALF_PATCH_SIZE,
            segmentation.shape[2]/2-HALF_PATCH_SIZE:segmentation.shape[2]/2+HALF_PATCH_SIZE
       ]

    assert clipped_mask.shape[-3:-1] == (PATCH_SIZE, PATCH_SIZE)
    assert clipped_image.shape[-3:-1] == (PATCH_SIZE, PATCH_SIZE)
    return clipped_image, clipped_mask


def segmentation_overlay(image, target, output):
    """
    """
    assert image.ravel().max() <= 1, f"Wrong tensor image intensity: {image.max()}."
    assert target.ravel().max() <= 1, f"Wrong tensor target intensity: {target.max()}."
    assert output.ravel().max() <= 1, f"Wrong tensor output intensity: {output.max()}."

    image = np.moveaxis(np.uint8(image.cpu().numpy()*255), 0, -1,)
    target = np.moveaxis(np.uint8(target.cpu().numpy()*255), 0, -1)
    output = np.moveaxis(np.uint8(output.cpu().numpy()*255), 0, -1)

    # edges overlay
    edges_gt = cv2.Canny(target, 0, 255)  # canny edge detector
    edges_seg = cv2.Canny(target, 0, 255)  # canny edge detector

    img = cv2.merge((image,image,image))  # creat RGB image from grayscale
    overlay = img.copy()
    overlay[edges_gt == 255] = [255, 0, 0]  # turn GT to red
    overlay[edges_seg == 255] = [0, 255, 0]  # turn segmentation edges to green

    # mask overlay
    masked_image = img.copy()

    masked_image_gt = np.where(
        target.astype(int),
        np.array([0,255,0], dtype='uint8'),
        masked_image)/255

    masked_image = np.where(
        output.astype(int),
        np.array([255,0,0], dtype='uint8'),
        masked_image)/255

    return torch.from_numpy(np.moveaxis(overlay/255, -1, 0)), \
        torch.from_numpy(np.moveaxis(cv2.addWeighted(masked_image_gt, 0.4, masked_image, 0.6, 0), -1, 0))
