from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "logger.log"
DATA_PATH = PROJECT_ROOT / "data"

# At least, the query must retrieve the following columns:
# ['ticker','exchcd','cusip','ncusip','comnam','permno','permco','namedt','nameendt','date']
STARTING_DATE = '1970-01-01' # of the wrds query YYYY-MM-DD
ENDING_DATE = '2019-12-31'
WRDS_REQUEST = """
WITH base AS (
    SELECT
        a.ticker, a.exchcd,
        a.comnam, a.cusip, a.ncusip,
        a.permno, a.permco,
        a.namedt, a.nameendt,
        b.date, b.ret, b.prc, b.shrout, b.vol,
        ABS(b.prc) * b.shrout * 1000 AS market_cap
    FROM crsp.msenames AS a
    JOIN crsp.dsf AS b
      ON a.permno = b.permno
     AND b.date BETWEEN a.namedt AND a.nameendt
    WHERE a.exchcd IN (1, 2, 3)          -- NYSE, AMEX, NASDAQ
      AND a.shrcd IN (10, 11)            -- Common shares only
      AND b.date >= '{starting_date}'
      AND b.date <= '{ending_date}'      -- optional: set an ending date for the query
      AND b.prc IS NOT NULL              -- ensure valid price
      AND b.vol IS NOT NULL              -- ensure valid volume
      AND b.prc != 0                     -- avoid zero-price issues
      AND ABS(b.prc) * b.vol >= 10000000 -- Dollar volume ≥ $10M
)
SELECT *
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY date ORDER BY market_cap DESC) AS mcap_rank
    FROM base
) ranked
WHERE mcap_rank <= 1000
ORDER BY date, mcap_rank;
"""

DATE_COLS = [
    'public_date'
]

SAVING_CONFIG_UNIVERSE = {
    'gross_query': {
        'path': DATA_PATH / "wrds_gross_query.parquet",
        'extension': 'parquet'
    },
    'universe': {
        'path': DATA_PATH / "wrds_universe.parquet",
        'extension': 'parquet'
    },
    'prices': {
        'path': DATA_PATH / "wrds_historical_prices.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_UNIVERSE = False
SAVING_CONFIG_PRICES = {
    'prices': {
        'path': DATA_PATH / "wrds_universe_prices.parquet",
        'extension': 'parquet'
    }
}
RETURN_BOOL_PRICES = False
RETURN_BOOL_RETURNS = False

