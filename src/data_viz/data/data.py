"""
A data manager class to handle the data for the data visualization project.
"""

import pandas as pd
from data_viz.utils.config import Config
import pandas as pd
from better_aws import AWS

class DataManager:
    """
    A class to manage the data for the data visualization project.
    """

    def __init__(self, config:Config) -> None:
        """
        Initialize the DataManager with the given data.
        :param config: a Config object containing the configuration for the data manager.
        :arg: dates: list of unique dates in the data, sorted in ascending order.
        :arg: universe: a dictionary mapping each date to a list of PERMNOs present on that date.
        """

        self.config = config
        self.funda_data: pd.DataFrame | None = None
        self.mkt_data: pd.DataFrame | None = None
        self.dates: list[pd.Timestamp] | None = None
        self.universe: dict[pd.Timestamp, list[int]] | None = None

    def load_data(self)->None:
        """
        Load the data from the source and process it to fill the attributes.
        :return:
        """
        self._connect_to_s3()
        self._fetch_from_s3()
        self._build_universe()

    def _connect_to_s3(self)->None:
        """
        Connect to the S3 bucket and load the data.
        :return:
        """
        self.aws = AWS(region=self.config.aws_default_region, verbose=True)
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.aws_bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
        )

    def _fetch_from_s3(self) -> None:
        """
        Fetch raw data from S3.

        :return: Dictionary containing raw dataframes loaded from S3.
        """
        self.funda_data = self.aws.s3.load(key=self.config.funda_path)
        self.mkt_data = self.aws.s3.load(key=self.config.mkt_path)

    def _build_universe(self)->None:
        """
        Build the universe attribute from the loaded data.
        :return:
        """
        self.dates = sorted(self.mkt_data['date'].unique().tolist())
        self.universe = (
            self.mkt_data
            .sort_values(['date', 'permno'])
            .groupby('date')['permno']
            .apply(list)
            .to_dict()
        )

