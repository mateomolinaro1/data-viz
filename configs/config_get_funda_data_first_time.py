from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "logger.log"
DATA_PATH = PROJECT_ROOT / "data"

# At least, the query must retrieve the following columns:
# ['ticker','exchcd','cusip','ncusip','comnam','permno','permco','namedt','nameendt','date']
STARTING_DATE = '1970-01-01' # of the wrds query YYYY-MM-DD
ENDING_DATE = '2019-12-31'

WRDS_REQUEST = """
SELECT *
FROM wrdsapps_finratio.firm_ratio
WHERE qdate >= DATE '{starting_date}'
    AND qdate <= DATE '{ending_date}'
ORDER BY qdate, ticker;
"""

DATE_COLS = [
    'namedt',
    'nameendt',
    'date'
]

SAVING_CONFIG_UNIVERSE = {
    'gross_query': {
        'path': DATA_PATH / "wrds_funda_gross_query.parquet",
        'extension': 'parquet'
    },
    'universe': {
        'path': DATA_PATH / "wrds_funda_universe.parquet",
        'extension': 'parquet'
    },
    'prices': {
        'path': DATA_PATH / "wrds_funda_historical_prices.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_UNIVERSE = False
SAVING_CONFIG_PRICES = {
    'prices': {
        'path': DATA_PATH / "wrds_funda_universe_prices.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_PRICES = False
RETURN_BOOL_RETURNS = False

