#!/usr/bin/env python3
"""Feature transformation pipeline for ML preprocessing.

Usage:
    python transform_features.py --input train.csv --output out.parquet \
        --config transforms.yaml --fit state.joblib
    python transform_features.py --input test.csv --output out.parquet --transform state.joblib
    python transform_features.py --input data.parquet --output features.parquet

YAML config (transforms.yaml):
    numerical:
      columns: [age, income]
      scaler: standard            # standard | minmax | robust
      log_transform: [income]     # apply log1p
      power_transform: []         # yeo-johnson
    categorical:
      columns: [gender, city]
      encoder: onehot             # onehot | label | ordinal | frequency
    text:
      columns: [description]
      method: tfidf               # tfidf | count
      max_features: 500
    datetime:
      columns: [created_at]
    imputation:
      strategy: median            # mean | median | mode | knn
"""
import argparse, logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(path):
    import pandas as pd
    p = str(path)
    if p.endswith(".parquet"): return pd.read_parquet(p)
    if p.endswith(".csv"): return pd.read_csv(p)
    if p.endswith(".json") or p.endswith(".jsonl"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    raise ValueError(f"Unsupported format: {p}")


def save_data(df, path):
    p = str(path)
    df.to_csv(p, index=False) if p.endswith(".csv") else df.to_parquet(p, index=False)
    logger.info(f"Saved {len(df)} rows x {len(df.columns)} cols -> {p}")


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def auto_detect_config(df):
    """Auto-detect column types and build default config."""
    import numpy as np
    cfg = {"numerical": {"columns": [], "scaler": "standard"},
           "categorical": {"columns": [], "encoder": "onehot"},
           "text": {"columns": [], "method": "tfidf", "max_features": 500},
           "datetime": {"columns": []}, "imputation": {"strategy": "median"}}
    for col in df.columns:
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            cfg["numerical"]["columns"].append(col)
        elif df[col].dtype == "datetime64[ns]" or "date" in col.lower():
            cfg["datetime"]["columns"].append(col)
        elif df[col].dtype == "object":
            if df[col].dropna().astype(str).str.len().mean() > 50:
                cfg["text"]["columns"].append(col)
            else:
                cfg["categorical"]["columns"].append(col)
    return cfg


def build_numerical_transformer(cfg):
    """Build sklearn Pipeline: impute -> optional log/power -> scale."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                       PowerTransformer, FunctionTransformer)
    from sklearn.impute import SimpleImputer
    import numpy as np
    imp = cfg.get("impute", "median")
    steps = [("imputer", SimpleImputer(strategy=imp if imp in ("mean", "median") else "median"))]
    if cfg.get("log_transform"):
        steps.append(("log", FunctionTransformer(np.log1p, validate=True)))
    if cfg.get("power_transform"):
        steps.append(("power", PowerTransformer(method="yeo-johnson")))
    scaler_map = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    steps.append(("scaler", scaler_map.get(cfg.get("scaler", "standard"), StandardScaler)()))
    return Pipeline(steps)


def build_categorical_transformer(cfg):
    """Build sklearn Pipeline: impute -> encode (onehot/ordinal/label)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    enc = cfg.get("encoder", "onehot")
    if enc in ("ordinal", "label"):
        steps.append(("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)))
    else:
        steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    return Pipeline(steps)


def build_text_transformer(cfg):
    """Build sklearn Pipeline: flatten column -> TF-IDF or CountVectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    max_f = cfg.get("max_features", 500)
    flatten = FunctionTransformer(lambda X: X.iloc[:, 0].fillna("").astype(str), validate=False)
    vec = (CountVectorizer(max_features=max_f) if cfg.get("method") == "count"
           else TfidfVectorizer(max_features=max_f))
    return Pipeline([("flatten", flatten), ("vectorizer", vec)])


def extract_datetime_features(df, columns):
    """Extract year/month/day/hour/dayofweek/is_weekend from datetime columns."""
    import pandas as pd
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        dt = pd.to_datetime(result[col], errors="coerce")
        for attr in ("year", "month", "day", "hour", "dayofweek"):
            result[f"{col}_{attr}"] = getattr(dt.dt, attr)
        result[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        result.drop(columns=[col], inplace=True)
        logger.info(f"Extracted 6 datetime features from '{col}'")
    return result


def apply_frequency_encoding(df, columns):
    """Replace categorical values with normalised frequency."""
    result, freq_maps = df.copy(), {}
    for col in columns:
        if col not in result.columns:
            continue
        freq = result[col].value_counts(normalize=True).to_dict()
        result[col] = result[col].map(freq).fillna(0.0)
        freq_maps[col] = freq
    return result, freq_maps


def apply_knn_imputation(df, numerical_cols):
    """Impute missing values with KNN imputer."""
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    cols = [c for c in numerical_cols if c in df.columns]
    if not cols:
        return df, None
    df[cols] = imputer.fit_transform(df[cols])
    logger.info(f"KNN imputation on {len(cols)} columns")
    return df, imputer


def build_column_transformer(config, df):
    """Build ColumnTransformer from config."""
    from sklearn.compose import ColumnTransformer
    transformers = []
    num_cfg = config.get("numerical", {})
    num_cols = [c for c in num_cfg.get("columns", []) if c in df.columns]
    if num_cols:
        transformers.append(("numerical", build_numerical_transformer(num_cfg), num_cols))
    cat_cfg = config.get("categorical", {})
    cat_cols = [c for c in cat_cfg.get("columns", []) if c in df.columns]
    if cat_cols and cat_cfg.get("encoder") != "frequency":
        transformers.append(("categorical", build_categorical_transformer(cat_cfg), cat_cols))
    text_cfg = config.get("text", {})
    for col in [c for c in text_cfg.get("columns", []) if c in df.columns]:
        transformers.append((f"text_{col}", build_text_transformer(text_cfg), [col]))
    return ColumnTransformer(transformers=transformers, remainder="passthrough",
                             verbose_feature_names_out=True) if transformers else None


def fit_and_transform(df, config):
    """Fit all transformers and return (transformed_df, state_dict)."""
    import pandas as pd
    state = {"config": config}
    df = extract_datetime_features(df, config.get("datetime", {}).get("columns", []))
    cat_cfg = config.get("categorical", {})
    if cat_cfg.get("encoder") == "frequency":
        cols = [c for c in cat_cfg.get("columns", []) if c in df.columns]
        df, state["freq_maps"] = apply_frequency_encoding(df, cols)
    if config.get("imputation", {}).get("strategy") == "knn":
        df, state["knn_imputer"] = apply_knn_imputation(
            df, config.get("numerical", {}).get("columns", []))
    ct = build_column_transformer(config, df)
    if ct is not None:
        transformed = ct.fit_transform(df)
        try:
            names = ct.get_feature_names_out()
        except Exception:
            names = [f"f_{i}" for i in range(transformed.shape[1])]
        result = pd.DataFrame(transformed, columns=names, index=df.index)
        state["column_transformer"] = ct
    else:
        result = df
    logger.info(f"Fit complete: {result.shape[0]} rows x {result.shape[1]} features")
    return result, state


def transform_only(df, state):
    """Apply previously fitted transformers to new data."""
    import pandas as pd
    config = state["config"]
    df = extract_datetime_features(df, config.get("datetime", {}).get("columns", []))
    for col, freq in state.get("freq_maps", {}).items():
        if col in df.columns:
            df[col] = df[col].map(freq).fillna(0.0)
    if "knn_imputer" in state and state["knn_imputer"] is not None:
        cols = [c for c in config.get("numerical", {}).get("columns", []) if c in df.columns]
        if cols:
            df[cols] = state["knn_imputer"].transform(df[cols])
    ct = state.get("column_transformer")
    if ct is not None:
        transformed = ct.transform(df)
        try:
            names = ct.get_feature_names_out()
        except Exception:
            names = [f"f_{i}" for i in range(transformed.shape[1])]
        result = pd.DataFrame(transformed, columns=names, index=df.index)
    else:
        result = df
    logger.info(f"Transform applied: {result.shape[0]} rows x {result.shape[1]} features")
    return result


def main():
    ap = argparse.ArgumentParser(description="Feature transformation pipeline")
    ap.add_argument("--input", required=True, help="Input data file (CSV/Parquet/JSON)")
    ap.add_argument("--output", required=True, help="Output file path (CSV/Parquet)")
    ap.add_argument("--config", default=None, help="YAML config for transforms")
    ap.add_argument("--fit", default=None, metavar="PATH", help="Fit and save state to joblib")
    ap.add_argument("--transform", default=None, metavar="PATH", help="Load state and apply")
    args = ap.parse_args()
    if args.fit and args.transform:
        logger.error("Cannot use --fit and --transform together."); sys.exit(1)
    logger.info(f"Loading data from {args.input}")
    df = load_data(args.input)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    if args.transform:
        import joblib
        result = transform_only(df, joblib.load(args.transform))
    else:
        config = load_config(args.config) if args.config else auto_detect_config(df)
        result, state = fit_and_transform(df, config)
        if args.fit:
            import joblib
            joblib.dump(state, args.fit)
            logger.info(f"State saved to {args.fit}")
    save_data(result, args.output)


if __name__ == "__main__":
    main()
