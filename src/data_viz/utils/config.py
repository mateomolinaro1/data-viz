from dataclasses import dataclass
from pathlib import Path
import logging
import json

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
        self.tbill_path: str | None = None  # DGS3MO (3-month T-bill), FRED format

        # Data
        # Network
        self.window_corr_matrix: int | None = None
        self.momentum_months: int | None = None
        self.layout_threshold_ref: float | None = None
        self.layout_sample_step: int | None = None
        self.layout_min_periods: int | None
        self.layout_seed: int | None = None
        self.label_market_cap_quantile: float | None = None
        self.min_edge_width: float | None = None
        self.max_edge_width: float | None = None
        self.min_node_opacity_fallback: float | None = None
        self.default_node_color: str | None = None
        self.positive_edge_color: str | None = None
        self.negative_edge_color: str | None = None
        self.node_spacing: int | None = None
        self.coordinate_scale: float | None = None

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

            if config.get("PATHS").get("S3_TBILL_PATH") is not None:
                self.tbill_path = config.get("PATHS").get("S3_TBILL_PATH")

            # Data
            if config.get("DATA").get("NETWORK").get("WINDOW_CORR_MATRIX") is not None:
                self.window_corr_matrix = config.get("DATA").get("NETWORK").get("WINDOW_CORR_MATRIX")

            if config.get("DATA").get("NETWORK").get("MOMENTUM_MONTHS") is not None:
                self.momentum_months = config.get("DATA").get("NETWORK").get("MOMENTUM_MONTHS")

            if config.get("DATA").get("NETWORK").get("LAYOUT_THRESHOLD_REF") is not None:
                self.layout_threshold_ref = config.get("DATA").get("NETWORK").get("LAYOUT_THRESHOLD_REF")

            if config.get("DATA").get("NETWORK").get("LAYOUT_SAMPLE_STEP") is not None:
                self.layout_sample_step = config.get("DATA").get("NETWORK").get("LAYOUT_SAMPLE_STEP")

            if config.get("DATA").get("NETWORK").get("LAYOUT_MIN_PERIODS") is not None:
                self.layout_min_periods = config.get("DATA").get("NETWORK").get("LAYOUT_MIN_PERIODS")

            if config.get("DATA").get("NETWORK").get("LAYOUT_SEED") is not None:
                self.layout_seed = config.get("DATA").get("NETWORK").get("LAYOUT_SEED")

            if config.get("DATA").get("NETWORK").get("LABEL_MARKET_CAP_QUANTILE") is not None:
                self.label_market_cap_quantile = config.get("DATA").get("NETWORK").get("LABEL_MARKET_CAP_QUANTILE")

            if config.get("DATA").get("NETWORK").get("MIN_EDGE_WIDTH") is not None:
                self.min_edge_width = config.get("DATA").get("NETWORK").get("MIN_EDGE_WIDTH")

            if config.get("DATA").get("NETWORK").get("MAX_EDGE_WIDTH") is not None:
                self.max_edge_width = config.get("DATA").get("NETWORK").get("MAX_EDGE_WIDTH")

            if config.get("DATA").get("NETWORK").get("MIN_NODE_OPACITY_FALLBACK") is not None:
                self.min_node_opacity_fallback = config.get("DATA").get("NETWORK").get("MIN_NODE_OPACITY_FALLBACK")

            if config.get("DATA").get("NETWORK").get("DEFAULT_NODE_COLOR") is not None:
                self.default_node_color = config.get("DATA").get("NETWORK").get("DEFAULT_NODE_COLOR")

            if config.get("DATA").get("NETWORK").get("POSITIVE_EDGE_COLOR") is not None:
                self.positive_edge_color = config.get("DATA").get("NETWORK").get("POSITIVE_EDGE_COLOR")

            if config.get("DATA").get("NETWORK").get("NEGATIVE_EDGE_COLOR") is not None:
                self.negative_edge_color = config.get("DATA").get("NETWORK").get("NEGATIVE_EDGE_COLOR")

            if config.get("DATA").get("NETWORK").get("NODE_SPACING") is not None:
                self.node_spacing = config.get("DATA").get("NETWORK").get("NODE_SPACING")

            if config.get("DATA").get("NETWORK").get("COORDINATE_SCALE") is not None:
                self.coordinate_scale = config.get("DATA").get("NETWORK").get("COORDINATE_SCALE")






