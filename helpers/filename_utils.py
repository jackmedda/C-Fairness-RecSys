import time
import secrets
from .constants import TIME_FORMATTER


def __random_id():
    return secrets.token_hex(5)


def __now():
    return time.strftime(TIME_FORMATTER, time.localtime())


def default_filepath():
    return f"{__now()}__{current_run_id()}"


def current_run_id():
    return __run_id


def set_run_id(run_id):
    global __run_id
    __run_id = run_id


__run_id = __random_id()
