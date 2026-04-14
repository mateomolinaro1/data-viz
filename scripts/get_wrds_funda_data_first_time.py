from configs.config_get_funda_data_first_time  import *
from dotenv import load_dotenv
load_dotenv()
import os
import wrds
import logging

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


wrds_db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"),
                          wrds_password=os.getenv("WRDS_PASSWORD"))

wrds_request = WRDS_REQUEST.format(starting_date=STARTING_DATE, ending_date=ENDING_DATE)
wrds_gross_query = wrds_db.raw_sql(sql=wrds_request,
                                   date_cols=['adate','qdate','public_date'])
wrds_gross_query.to_parquet(SAVING_CONFIG_UNIVERSE['gross_query']['path'], index=False)
wrds_db.close()
