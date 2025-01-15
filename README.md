# Medloader project

A dataloader to prepare and load any Medical image dataset, prepared to deal with .nii or DICOM formats.

### Usage:
```
from medloader.api import DatasetEngine
from medloader.app import config as config
from medloader.app.logger import init_logger
from torch.utils.data import DataLoader


logger = init_logger()
batch_size = 8

cfg = config.Configuration().get("datasets")

engine = DatasetEngine(
    logger=logger,
    config=cfg,
).go(mode="config")


train_dl = DataLoader(
    dataset=engine[0],
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    #sampler=DistributedSampler(dataset)
)
```

### Extending your own medical dataset:
```
from .medicaldataloader import MedicalDataloader

class NEWDataloader(MedicalDataloader):

    def dataset_clipping_preparation: # Optional
    def image_data_stats:
    def transform_processor: 
```

