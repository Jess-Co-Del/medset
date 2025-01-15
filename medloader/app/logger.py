
import logging


def init_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s: %(message)s"
    )

    logger = logging.getLogger("App")
    return logger
