import logging
import os
from os.path import join
from pathlib import Path

import graypy

APP_PATH = Path(__file__).parent.parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
DATA_PATH = join(ROOT_PATH, "models_data")
GRAYLOG_IP = os.environ.get("GRAYLOG_IP")

HUGGINGFACE_PATH = join(ROOT_PATH, "huggingface")

handlers = [logging.StreamHandler()]

if GRAYLOG_IP:
    handlers.append(graypy.GELFUDPHandler(GRAYLOG_IP, 12201, localname="pdf_metadata_extraction"))

logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
config_logger = logging.getLogger(__name__)
