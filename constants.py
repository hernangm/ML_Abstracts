from dataclasses import dataclass

__body__ = "Body"
__target__ = "AverageScore"

@dataclass(frozen=True)
class DataSetConfig:
    PATH: str = "_dataset\\Abstracts.xlsx"
    FEATURES_BODY = __body__
    FEATURES_TARGET = __target__
    COLUMNS = ["Title", __body__, "Scores", __target__]
