"""class DataManager:
Attribut 1 appelé date : liste de date du DataFrame wrdsdata
Attribut 2 universe sous forme de dictionnaire
en clef date associé liste une valeur permno"""

import pandas as pd
from typing import Optional




class DataManager:
    """
    Attributs
    ---------
    dates : list[pd.Timestamp]
        Liste des dates uniques du DataFrame.
    universe : dict[pd.Timestamp, list[int]]
        Pour chaque date, la liste des PERMNO présents.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Attribut 1 — liste des dates uniques triées
        self.dates: list[pd.Timestamp] = sorted(data['date'].unique().tolist())

        # Attribut 2 — dictionnaire date → liste de PERMNO
        self.universe: dict[pd.Timestamp, list[int]] = (
            data.groupby('date')['permno']
            .apply(list)
            .to_dict()
        )


