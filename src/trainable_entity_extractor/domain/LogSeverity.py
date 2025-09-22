from enum import Enum


class LogSeverity(str, Enum):
    error = "error"
    info = "info"
    warning = "warning"
