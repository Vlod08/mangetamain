from __future__ import annotations
import pandas as pd
from .filters import parse_tag_str

COUNTRY_KEYWORDS = {
    "italian": "Italy",
    "italy": "Italy",
    "french": "France",
    "france": "France",
    "mexican": "Mexico",
    "mexico": "Mexico",
    "indian": "India",
    "china": "China",
    "chinese": "China",
    "japanese": "Japan",
    "japan": "Japan",
    "spanish": "Spain",
    "spain": "Spain",
    "greek": "Greece",
    "moroccan": "Morocco",
    "thai": "Thailand",
    "turkish": "Türkiye",
    "lebanese": "Lebanon",
    "american": "USA",
    "usa": "USA",
    "british": "UK",
    "english": "UK",
    "german": "Germany",
}


def infer_country(tags_val: str | list | None) -> str | None:
    tags = parse_tag_str(tags_val) if not isinstance(tags_val, list) else tags_val
    tags = [t.lower() for t in tags]
    for t in tags:
        if t in COUNTRY_KEYWORDS:
            return COUNTRY_KEYWORDS[t]
    return None


def add_country_column(df: pd.DataFrame) -> pd.DataFrame:
    if "country" not in df.columns:
        df["country"] = df["tags"].apply(infer_country)
    return df
