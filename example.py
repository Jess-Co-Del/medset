from medloader.api import DatasetEngine
from medloader.app import config as config
from medloader.app.logger import init_logger

logger = init_logger()

cfg = config.Configuration()

engine = DatasetEngine(
    logger=logger,
    config=cfg,
).go(mode="setup_clipping")

# engine.scans_to_image_files()

# engine.go()

# from monai.data import GridPatchDataset, DataLoader, PatchIter
# import numpy as np
# import torch
# from monai.data.utils import iter_patch

# images = torch.randn([1,1,4,4])

# patches = iter_patch(images, patch_size=0, start_pos=(0, 0), overlap=0.0, copy_back=True)