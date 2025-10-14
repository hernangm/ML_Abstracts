from dataclasses import dataclass

__body__ = "Body"
__target__ = "AverageScore"
__title__ = "Title"

@dataclass(frozen=True)
class DataSetConfig:
    PATH: str = "data/processed/Abstracts.xlsx"
    TITLE: str = __title__
    FEATURES_BODY: str = __body__
    FEATURES_TARGET: str = __target__
    COLUMNS = [__title__, __body__, "Topic_1", "Presentation_Type", __target__]
