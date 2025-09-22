import logging
import os
from os.path import join
from pathlib import Path

import graypy

APP_PATH = Path(__file__).parent.parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
DATA_PATH = Path(ROOT_PATH, "models_data")
CACHE_PATH = Path(DATA_PATH, "cache")
EXTRACTOR_JOB_PATH = Path("extractor_job", "extractor_job.json")
GRAYLOG_IP = os.environ.get("GRAYLOG_IP")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_PATH = join(ROOT_PATH, "huggingface")

IS_TRAINING_CANCELED_FILE_NAME = "is_training_canceled.txt"

handlers = [logging.StreamHandler()]

if GRAYLOG_IP:
    handlers.append(graypy.GELFUDPHandler(GRAYLOG_IP, 12201, localname="pdf_metadata_extraction"))

logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
config_logger = logging.getLogger(__name__)
