from configs.config_get_wrds_mkt_data_first_time  import *
from data_viz.data.data_handler import DataHandler
from dotenv import load_dotenv
load_dotenv()
import os
import logging

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

dh = DataHandler(data_path=DATA_PATH,
                 wrds_username=os.getenv("WRDS_USERNAME"),
                 wrds_password=os.getenv("WRDS_PASSWORD")
                 )

dh.connect_wrds()
dh.fetch_wrds_historical_universe(wrds_request=WRDS_REQUEST,
                                  starting_date=STARTING_DATE,
                                  ending_date=ENDING_DATE,
                                  date_cols=DATE_COLS,
                                  saving_config=SAVING_CONFIG_UNIVERSE,
                                  save_tickers_across_dates=True,
                                  save_dates=True,
                                  return_bool=RETURN_BOOL_UNIVERSE)
dh.get_wrds_historical_prices(saving_config=SAVING_CONFIG_PRICES)
dh.get_wrds_returns()
dh.logout_wrds()
logging.info("Data fetching and processing completed.")

import pandas as pd
a = pd.read_parquet(r"C:\Users\mateo\Code\ENSAEDataVisualization\DataViz\data\wrds_gross_query.parquet")
b = pd.read_parquet(r"C:\Users\mateo\Code\ENSAEDataVisualization\DataViz\data\wrds_gross_query1.parquet")
c = pd.concat([a,b], axis=0)
del a
del b
c.to_parquet(r"C:\Users\mateo\Code\ENSAEDataVisualization\DataViz\data\wrds_gross_query.parquet")