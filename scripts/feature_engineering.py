import pandas as pd
from data_viz.utils.config import Config
config = Config()

wrds_mkt_data = pd.read_parquet(
    config.ROOT_DIR / "data" / "wrds_gross_query.parquet"
)
wrds_funda_data = pd.read_parquet(
    config.ROOT_DIR / "data" / "wrds_funda_gross_query.parquet"
)