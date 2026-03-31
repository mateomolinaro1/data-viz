from data_viz.utils.config import Config
from data_viz.data.data import DataManager
from dotenv import load_dotenv
import sys
import logging

# Init
load_dotenv()
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()

# Data
dm = DataManager(config=config)
dm.load_data()
