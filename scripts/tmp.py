import pandas as pd
from data_viz.utils.config import Config
from data_viz.data.data import DataManager
from data_viz.network.network import NetworkBuilder
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
dm.build_reference_network_layout()

date = dm.dates[-1]
threshold = 0.65
min_periods = dm.config.layout_min_periods

nodes1, edges1 = dm.get_graph_snapshot(date, threshold, min_periods)
nodes2, edges2 = dm.get_graph_snapshot(date, threshold, min_periods)

print(nodes1.shape, edges1.shape)
print(nodes2.shape, edges2.shape)

stats = dm.get_graph_summary_stats(date, threshold, min_periods)
print(stats)

ranking = dm.get_node_ranking_table(date, threshold, min_periods)
print(ranking.head())