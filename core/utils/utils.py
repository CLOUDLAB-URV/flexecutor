import logging


def setup_logging(level):
    # Loggers
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    debug_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s"
    info_format = "%(asctime)s [%(levelname)s] %(message)s"

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(debug_format, datefmt="%Y-%m-%d %H:%M:%S")
    debug_handler.setFormatter(debug_formatter)

    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter(info_format, datefmt="%Y-%m-%d %H:%M:%S")
    info_handler.setFormatter(info_formatter)

    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.propagate = False

    return logger
