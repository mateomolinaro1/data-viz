from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import List, Dict, Type, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
    """
    def __init__(self):
        # Paths
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()
        logger.info("Root dir: " + str(self.ROOT_DIR))

        self.RUN_PIPELINE_CONFIG_PATH = self.ROOT_DIR / "configs" / "run_pipeline_config.json"
        logger.info("run_pipeline config path: " + str(self.RUN_PIPELINE_CONFIG_PATH))

        # AWS profile
        self.aws_bucket_name: str | None = None
        self.aws_default_region: str | None = None
        self.aws_output_format: str | None = None

        # Data path configurations
        self.funda_path: str | None | Path = None
        self.mkt_path: str | None | Path = None
        self.s3_path: str | None | Path = None

        # Load json config to attributes of Config class
        self._load_run_pipeline_config()

    def _load_run_pipeline_config(self)->None:
        """
        Load run_pipeline_config.json file
        :return:
        """
        with open(self.ROOT_DIR / "configs" / "run_pipeline_config.json" , "r") as f:
            config: dict = json.load(f)

            # AWS
            if config.get("AWS").get("BUCKET_NAME") is not None:
                self.aws_bucket_name = config.get("AWS").get("BUCKET_NAME")
            if config.get("AWS").get("DEFAULT_REGION") is not None:
                self.aws_default_region = config.get("AWS").get("DEFAULT_REGION")
            if config.get("AWS").get("OUTPUT_FORMAT") is not None:
                self.aws_output_format = config.get("AWS").get("OUTPUT_FORMAT")

            # Paths
            if config.get("PATHS").get("S3_FUNDA_PATH") is not None:
                self.funda_path = config.get("PATHS").get("S3_FUNDA_PATH")

            if config.get("PATHS").get("S3_MKT_PATH") is not None:
                self.mkt_path = config.get("PATHS").get("S3_MKT_PATH")

            if config.get("PATHS").get("S3_PATH") is not None:
                self.s3_path = config.get("PATHS").get("S3_PATH")


