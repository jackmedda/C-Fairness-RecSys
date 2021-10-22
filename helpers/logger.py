import logging
import os
from typing import Sequence, Tuple, Callable

import helpers.filename_utils as file_utils
from .constants import (
    LOGGER_NAME,
    LOGS_PATH,
    TIME_FORMATTER,
    LOG_STREAM_FORMATTER,
    LOG_FILE_FORMATTER,
    LOG_ERROR_FORMATTER
)


class RcLogger(object):

    _rclogger_name = None

    @staticmethod
    def get(logger_name=None):
        logger_name = RcLogger._rclogger_name if logger_name is None else logger_name

        if logger_name is None:
            raise AttributeError("Logger has not been started")

        return logging.getLogger(logger_name)

    @staticmethod
    def start_logger(logger_name=LOGGER_NAME,
                     level="DEBUG",
                     init_logging: Sequence[Tuple[str, str]] = None,
                     file_log=True):
        RcLogger._rclogger_name = logger_name

        session_id = "session_id"
        if init_logging is None:
            init_logging = [session_id]

        logger = RcLogger.get(RcLogger._rclogger_name)
        logger.propagate = 0

        logger_filepath = os.path.join(LOGS_PATH, file_utils.default_filepath() + '.log')

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(fmt=LOG_STREAM_FORMATTER, style='{')
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(level.upper())
        logger.addHandler(stream_handler)

        if file_log:
            file_handler = logging.FileHandler(logger_filepath)
            file_formatter = logging.Formatter(fmt=LOG_FILE_FORMATTER, datefmt=TIME_FORMATTER, style='{')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel("DEBUG")
            logger.addHandler(file_handler)

        logger.setLevel("DEBUG")

        first_logs = [
            ("debug", f"Logger initialized with stream handler and file handler at path '{logger_filepath}'")
        ]
        if session_id in init_logging:
            init_logging.remove(session_id)
            first_logs = [session_id] + first_logs

        init_logging = first_logs + init_logging

        for init_log in init_logging:
            if init_log == session_id:
                logger.info(f"Run session id: {file_utils.current_run_id()}")
            else:
                init_log: Tuple[str, str]  # type hint

                log_function: Callable = getattr(logger, init_log[0].lower())
                log_function(init_log[1])


class RcLoggerException(Exception):
    """Report a logger.error before raising the Exception"""

    def __init__(self, exc, msg):
        RcLogger.get().error(LOG_ERROR_FORMATTER.format(exc=exc, msg=msg))
        raise exc(msg)
