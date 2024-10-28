import logging
from os.path import join
from pathlib import Path


APP_PATH = Path(__file__).parent.parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
DATA_PATH = join(ROOT_PATH, "models_data")

HUGGINGFACE_PATH = join(ROOT_PATH, "huggingface")

handlers = [logging.StreamHandler()]

logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
config_logger = logging.getLogger(__name__)
