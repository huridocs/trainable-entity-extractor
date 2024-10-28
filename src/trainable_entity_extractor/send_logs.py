import traceback

from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LogsMessage import Severity


def send_logs(
    extraction_identifier: ExtractionIdentifier,
    message: str,
    severity: Severity = Severity.info,
    exception: Exception = None,
):
    if severity != Severity.error:
        config_logger.info(message + " for " + extraction_identifier.model_dump_json())
        return

    try:
        stacktrace_message = "\n".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        error_message = message
        error_message += f"\nException type: {type(exception).__name__}"
        error_message += f"\nException: {exception}"
        error_message += f"\nStackTrace: {stacktrace_message}"
        config_logger.error(error_message + " for " + extraction_identifier.model_dump_json())
    except:
        config_logger.error(message + " for " + extraction_identifier.model_dump_json())
