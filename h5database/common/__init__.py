from .split import *
import sys


def get_path_limit():
    LIMIT_WIN = 260
    LIMIT_LINUX = 4096
    os_name = sys.platform
    if os_name == "win32":
        return LIMIT_WIN
    return LIMIT_LINUX
