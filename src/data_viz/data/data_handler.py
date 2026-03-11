import pandas as pd
import pickle
from typing import List, Union
import logging
import wrds
from pathlib import Path


logger = logging.getLogger(__name__)

class DataHandler:
    """
    A class to handle data fetching and processing from WRDS and Interactive Brokers (IB).
    It requires to have the IB Gateway or TWS running and configured to accept API connections (right host
    and port, which you can find in the IB Gateway settings).
    More info: https://www.interactivebrokers.eu/campus/ibkr-api-page/twsapi-doc/#api-introduction
    Attributes:
    - wrds_username: WRDS username for database connection.
    - ib_host: Host address for IB API connection.
    - ib_port: Port number for IB API connection.
    - ib_client_id: Client ID for IB API connection.
    - pause: Pause duration between IB API requests to avoid rate limiting.
    - retries: Number of retries for IB API requests in case of failure.
    - reconnect_wait: Wait time before reconnecting to IB API after a disconnection.
    -
    Methods:
    """
    def __init__(self,
                 data_path: Path,
                 wrds_username:str,
                 wrds_password:str
    )->None:
        """
        Initializes the DataHandler with WRDS and IB connection parameters.
        :parameters:
        - data_path: path of the data folder.
        - wrds_username: WRDS username for database connection.
        - ib_host: Host address for IB API connection.
        - ib_port: Port number for IB API connection.
        - ib_client_id: Client ID for IB API connection.
        - pause: Pause duration between IB API requests to avoid rate limiting.
        - retries: Number of retries for IB API requests in case of failure.
        - reconnect_wait: Wait time before reconnecting to IB API after a disconnection.
        - wrds_db: WRDS database connection object.
        - ib: IB API connection object.
        - wrds_gross_query: Raw WRDS query result DataFrame.
        - crsp_to_ib_mapping_tickers: Dictionary mapping CRSP tickers to IB tickers.
        - wrds_universe: Processed WRDS universe DataFrame.
        - wrds_universe_last_date: Last date in the WRDS universe DataFrame.
        - universe_prices_wrds: WRDS universe prices DataFrame.
        - universe_returns_wrds: WRDS universe returns DataFrame.
        - fields_wrds_to_keep_for_universe: List of fields to keep in the WRDS universe DataFrame.
        - crsp_to_ib_exchange: Dictionary mapping CRSP exchange codes to IB exchange names.
        - crps_exchcd_to_currency: Dictionary mapping CRSP exchange codes to currency codes.
        - tickers_across_dates: List of unique tickers across all dates in the WRDS universe.
        - ib_tickers: List of IB tickers corresponding to CRSP tickers.
        - dates: List of unique dates in the WRDS universe.
        - universe_prices_ib: IB universe prices DataFrame.
        - universe_returns_ib: IB universe returns DataFrame.
        - valid_tickers_per_ib_date: Series of valid tickers per IB date.
        - valid_permnos_per_ib_date: Series of valid PERMNOs per IB date.
        """
        self.data_path = data_path
        self.wrds_username = wrds_username
        self.wrds_password = wrds_password

        self.wrds_db = None

        self.wrds_gross_query = None
        self.wrds_universe = None
        self.wrds_universe_last_date = None
        self.universe_prices_wrds = None
        self.universe_returns_wrds = None
        self.fields_wrds_to_keep_for_universe = ['ticker',
                                                 'exchcd',
                                                 'cusip',
                                                 'ncusip',
                                                 'comnam',
                                                 'permno',
                                                 'permco',
                                                 'namedt',
                                                 'nameendt',
                                                 'date']

        self.tickers_across_dates = None
        self.dates = None

    def connect_wrds(self):
        """Establishes a connection to the WRDS database using the provided username."""
        self.wrds_db = wrds.Connection(wrds_username=self.wrds_username,
                                       wrds_password=self.wrds_password)

    def logout_wrds(self):
        """Logs out from the WRDS database connection."""
        if self.wrds_db is not None:
            self.wrds_db.close()
            self.wrds_db = None

    def fetch_wrds_historical_universe(self,
                                       wrds_request:str,
                                       starting_date:str,
                                       date_cols:List[str],
                                       saving_config:dict,
                                       save_tickers_across_dates:bool=True,
                                       save_dates:bool=True,
                                       return_bool:bool=False
                                       )->Union[None,dict]:
        """
        Fetches historical universe from WRDS based on the provided SQL request. It saves wrds_gross_query
        and wrds_universe to disk if specified in saving_config.
        :parameters:
        - wrds_request: SQL query string to fetch data from WRDS.
        - date_cols: List of columns in the query result that should be parsed as dates.
        - saving_config: Dictionary specifying saving paths and formats for gross query and universe.
        - return_bool: If True, returns the fetched universe DataFrame.
        """
        # Check input data types
        if not isinstance(wrds_request, str):
            logger.error("wrds_request must be a string.")
            raise ValueError("wrds_request must be a string containing the SQL query.")
        if not isinstance(starting_date, str):
            logger.error("starting_date must be a string.")
            raise ValueError("starting_date must be a string in 'YYYY-MM-DD' format.")
        if not isinstance(date_cols, list):
            logger.error("date_cols must be a list of strings.")
            raise ValueError("date_cols must be a list of strings.")
        for col in date_cols:
            if not isinstance(col, str):
                logger.error("All elements in date_cols must be strings.")
                raise ValueError("All elements in date_cols must be strings.")
        if not isinstance(saving_config, dict):
            logger.error("saving_config must be a dictionary.")
            raise ValueError("saving_config must be a dictionary.")
        if not isinstance(return_bool, bool):
            logger.error("return_bool must be a boolean.")
            raise ValueError("return_bool must be a boolean.")

        # Ensure connection to WRDS
        if self.wrds_db is None:
            self.connect_wrds()

        # Query WRDS database
        wrds_request = wrds_request.format(starting_date=starting_date)
        self.wrds_gross_query = self.wrds_db.raw_sql(sql=wrds_request,
                                                     date_cols=date_cols)

        # As unique identifiers of WRDS/CRSP are PERMNO and not ticker but for IB it is ticker,
        # We have to ensure (date, permno) are unique, and then we will have to create a mapping
        # between CRSP tickers and IB tickers
        self.wrds_gross_query = self.wrds_gross_query.drop_duplicates(subset=['date', 'permno'],
                                                                      keep='last')
        # Sort for checking
        self.wrds_gross_query = self.wrds_gross_query.sort_values(by=['date'],
                                                                  ascending=True).reset_index(drop=True)
        self.tickers_across_dates = list(self.wrds_gross_query['ticker'].unique())
        if save_tickers_across_dates:
            with open(self.data_path / "tickers_across_dates.pkl", "wb") as f:
                pickle.dump(self.tickers_across_dates, f)

        self.dates = list(self.wrds_gross_query['date'].unique())
        if save_dates:
            with open(self.data_path / "dates.pkl", "wb") as f:
                pickle.dump(self.dates, f)

        # Save gross query if specified
        if 'gross_query' in saving_config:
            if saving_config['gross_query']['extension'] == 'parquet':
                self.wrds_gross_query.to_parquet(saving_config['gross_query']['path'],
                                                index=False)
            else:
                logger.error("Unsupported file extension for gross query.")
                raise ValueError("Unsupported file extension for gross query. Use 'parquet'.")


        universe = self.wrds_gross_query.copy()
        universe.index = universe['date']
        self.wrds_universe = universe


        # Save to file if a saving path is provided
        if 'universe' in saving_config:
            if saving_config['universe']['extension'] == 'parquet':
                self.wrds_universe.to_parquet(saving_config['universe']['path'],
                                              index=True)
            else:
                logger.error("Unsupported file extension for universe.")
                raise ValueError("Unsupported file extension for universe. Use 'parquet'.")

        if return_bool:
            return {'wrds_gross_query':self.wrds_gross_query,
                    'wrds_universe':self.wrds_universe}
        return None

    def get_wrds_historical_prices(self,
                                   saving_config:dict,
                                   return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Format self.wrds_gross_query to have a nice prices df.
        :parameters:
        - saving_config: Dictionary specifying saving paths and formats for prices.
        - return_bool: If True, returns the prices DataFrame.
        It either saves the prices DataFrame to disk or returns it based on the parameters.
        """
        if self.wrds_gross_query is None:
            try:
                self.wrds_gross_query = pd.read_parquet(self.data_path / "wrds_gross_query.parquet")
            except Exception as e:
                logger.error(f"Error reading WRDS gross query: {e}")
                raise ValueError("WRDS universe data is not loaded. Please fetch it first.")

        prices = self.wrds_gross_query.pivot(values='prc',
                                             index='date',
                                             columns='permno')
        self.universe_prices_wrds = prices

        if 'prices' in saving_config:
            if saving_config['prices']['extension'] == 'parquet':
                prices.to_parquet(saving_config['prices']['path'],
                              index=True)
            else:
                raise ValueError("Unsupported file extension for prices. Use 'parquet'.")

        if return_bool:
            return prices
        return None

    def get_wrds_returns(self,
                         return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Compute returns DataFrame from universe prices
        :parameters:
        - return_bool: If True, returns the returns DataFrame.
        """
        if self.universe_prices_wrds is None:
            raise ValueError("Universe prices data is not loaded. Please fetch it first.")
        returns = self.universe_prices_wrds.pct_change(fill_method=None)
        self.universe_returns_wrds = returns
        if return_bool:
            return returns
        return None
