import os
import traceback

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity


def send_logs(
    extraction_identifier: ExtractionIdentifier,
    message: str,
    severity: LogSeverity = LogSeverity.info,
    exception: Exception = None,
):
    machine_name = os.uname().nodename
    if severity != LogSeverity.error:
        config_logger.info(message + " for " + extraction_identifier.model_dump_json() + " on " + machine_name)
        return

    try:
        stacktrace_message = "\n".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        error_message = message
        error_message += f"\nException type: {type(exception).__name__}"
        error_message += f"\nException: {exception}"
        error_message += f"\nStackTrace: {stacktrace_message}"
        config_logger.error(error_message + " for " + extraction_identifier.model_dump_json() + " on " + machine_name)
    except:
        config_logger.error(message + " for " + extraction_identifier.model_dump_json() + " on " + machine_name)
