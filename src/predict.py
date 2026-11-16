"""Predicting new, label-free rows with the refit pipeline."""
import os

import pandas as pd

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new(pipeline, config: dict, feature_columns: list) -> str:
    """
    Predicts the rows in `config["predict"]["path"]` with the refit `pipeline` and writes one
    row per input row (id, probability of class 1, 0/1 prediction at `threshold`) to
    `config["predict"]["output"]`. The data gets the same cleaning and feature split as the
    training data, but no de-duplication: every row needs a prediction. Returns the path.
    """
    predict_config = config["predict"]
    threshold = predict_config.get("threshold", 0.5)
    id_column = config["diagnostics"].get("id_column")

    df_new = load_data(predict_config["path"])
    df_clean = clean_dataset(df_new, config["diagnostics"])
    X_new, _, _ = split_features_target(df_clean, config["data"],
                                        config["preprocessing"].get("mnar_indicator_sources", []))
    X_new = X_new.reindex(columns=feature_columns)          # same columns, same order as training

    proba = pipeline.predict_proba(X_new)[:, 1]
    ids = df_clean[id_column].values if id_column and id_column in df_clean.columns else df_clean.index
    out = pd.DataFrame({id_column or "row": ids, "probability": proba,
                        "prediction": (proba >= threshold).astype(int)})
    if len(out) != len(df_new):
        raise ValueError(f"{len(df_new)} rows in, but {len(out)} predictions out")

    path = predict_config["output"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path
