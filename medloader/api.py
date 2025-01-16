"""
Interactive class API with the Dataset generators
"""
from enum import Enum
from torch.utils.data import ConcatDataset

from . import IrcadbDataloader, LitsDataloader


class DatasetStore(Enum):
    """ Enumeration for the implemented dataset objects"""

    IRCAD = "ircadb"
    LITS = "Lits"


class DatasetEngine:
    """ Abstraction from the Datasets implemented for medical image processing with
    Pytorch.
    """
    def __init__(
        self,
        logger,
        config
    ):

        self.logger = logger
        self.config = config

    # @classmethod
    def from_dataset(
        self,
        algorithm: DatasetStore,
        transform: bool,
        verbose: bool,
        config: dict
    ) -> "AlgorithmEngine":
        """
        Instantiates a concrete implementation of the Algorithm from its enumeration.

        Parameters:
        algorithm (DatasetStore): The algorithm enumeration
        transform (dict): The dictionary of transforms to apply to a train dataset.
        """

        if algorithm == DatasetStore.IRCAD.value:
            return IrcadbDataloader(
                logger=self.logger,
                transform=transform,
                verbose=verbose,
                cubic_slicer_context_layers=self.config["cubic_slicer_context_layers"],
                **config
            )

        elif algorithm == DatasetStore.LITS.value:
            return LitsDataloader(
                logger=self.logger,
                transform=transform,
                verbose=verbose,
                cubic_slicer_context_layers=self.config["cubic_slicer_context_layers"],
                **config
            )

        else:
            raise NotImplementedError()

    def go(
        self,
        mode: str = "config",
        ):
        """
        Private function that builds a set of datasets, tailored via config file.

        1. Prepares a "train" and "validation/test" Dataset objects
        2. Prepared to receive more than one dataset to "train".
        3. Prepared to load from hard config, or in cross-validation mode preparares
        "validation" dataset from "train" config.

        :param validation_folder_sample: Parameter provides list of folder samples
            to be used as validation. If it is None, it assumes folder_samples
            provided via config file. Otherwise we are in Cross-Validation mode.
        :returns: A dictionary of the loaded and transformed dictionaries.
        """
        train_dataset_objects = dict()

        if mode == "setup_processed":  # process raw .nii data files.
            return self.from_dataset(
                algorithm=list(self.config["train"].keys())[0],
                transform=False,
                verbose=False,
                config=self.config["train"][list(self.config["train"].keys())[0]]
            ).volume_raw_processing()

        elif mode == "setup_clipping":  # process patched/clipped dataset
            return self.from_dataset(
                algorithm=list(self.config["train"].keys())[0],
                transform=False,
                verbose=False,
                config=self.config["train"][list(self.config["train"].keys())[0]]
            ).dataset_clipping()

        elif mode == "config":  # Hard config mode for 'processed' dataset train/test load.
            train_dataset_objects = []
            print(list(self.config["train"].keys()))
            for dataset_object in list(self.config["train"].keys()):
                for _ in range(1):  ## SECOND appends transformed dataset(S)
                    train_dataset_objects.append(
                        self.from_dataset(
                            algorithm=dataset_object,
                            transform=True,
                            verbose=True,
                            config=self.config["train"][dataset_object]
                        )
                    )
                train_dataset_objects.append(
                    self.from_dataset(
                        algorithm=dataset_object,
                        transform=False,  ## FIRST appends original dataset
                        verbose=True,
                        config=self.config["train"][dataset_object]
                    )
                )

            train_dataset = ConcatDataset(train_dataset_objects)

            val_dataset = self.from_dataset(
                algorithm=list(self.config["val"].keys())[0],
                transform=False,
                verbose=True,
                config=self.config["val"][list(self.config["val"].keys())[0]]
            )

        return train_dataset, val_dataset
