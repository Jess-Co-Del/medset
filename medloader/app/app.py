"""
Medical image Semantic System Framework orchestrator.

"""

from . import config
# from algorithmEngine import AlgorithmEngine, Algorithm
from . import api
from .logger import init_logger

logger = init_logger()


class Application:
    """
    Class used to represent the Segmentation System Framework.

    Attributes
    ------------------------------
    configuration : dict (optional)
        the configuration object used to overwrite the default configurations

    Methods
    ------------------------------
"""
    def __init__(self, configuration_dict: dict = None):

        self.config = config.Configuration(configuration_dict)

        logger.debug("Created SSNetwork System Application.")

    def data_runner(self):
        """
        """

        method = api.ModelTrainer(logger=logger, config=self.config)

    def model_trainer(
        self,
    ):
        """
        """
        method = api.ModelTrainer(logger=logger, config=self.config)

        return method.go()

    def model_inference(
            self,
    ):
        """
        """
        method = api.ModelInference(logger=logger, config=self.config)

        return method.go()

    def model_cross_validate(
            self,
    ):
        """
        """
        method = api.ModelCrossValidator(logger=logger, config=self.config)

        return method.go(custom_run=True)

    def model_cross_val_inference(
            self,
    ):
        """
        """
        method = api.ModelCrossValInference(logger=logger, config=self.config)

        return method.go(custom_run=True)
