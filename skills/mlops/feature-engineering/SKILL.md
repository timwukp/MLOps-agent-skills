---
name: feature-engineering
description: >
  Design and implement feature engineering pipelines for ML models. Covers numerical transformations (scaling,
  normalization, binning, polynomial), categorical encoding (one-hot, target, ordinal, hashing), text features
  (TF-IDF, embeddings, tokenization), image features (CNN extraction, augmentation), time-series features (lag,
  rolling, Fourier), missing value imputation (KNN, MICE), feature selection (mutual info, SHAP, RFE, L1),
  automated feature engineering (Featuretools, tsfresh), scikit-learn Pipelines, ColumnTransformer, dimensionality
  reduction (PCA, UMAP), and production feature pipeline best practices.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Feature Engineering for ML

## Overview

Feature engineering transforms raw data into informative features that improve model performance.
It is often the highest-leverage activity in an ML project.

## When to Use This Skill

- Building feature pipelines for new ML models
- Improving model performance through better features
- Handling new data types (text, images, time-series)
- Setting up reproducible feature transformations
- Selecting the most impactful features

## Step-by-Step Instructions

### 1. Numerical Transformations

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, KBinsDiscretizer
)
import numpy as np

# Standard scaling (mean=0, std=1) - default choice
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[["age", "income"]])

# Min-Max scaling [0, 1] - for neural networks, distance-based models
scaler = MinMaxScaler()

# Robust scaling - resistant to outliers (uses median/IQR)
scaler = RobustScaler()

# Log transform - for right-skewed distributions
X["log_income"] = np.log1p(X["income"])

# Power transform (Box-Cox/Yeo-Johnson) - make data more Gaussian
transformer = PowerTransformer(method="yeo-johnson")

# Quantile transform - uniform or normal output distribution
transformer = QuantileTransformer(output_distribution="normal")

# Binning - convert continuous to categorical
binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
```

**When to use which scaler:**

| Method | Use When |
|--------|----------|
| StandardScaler | Default; linear models, SVMs |
| MinMaxScaler | Neural networks, bounded features |
| RobustScaler | Data has outliers |
| PowerTransformer | Skewed distributions |
| QuantileTransformer | Need uniform/normal distribution |
| Log transform | Right-skewed (income, counts) |

### 2. Categorical Encoding

```python
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder

# One-hot encoding - for low cardinality (< 15 categories)
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[["color", "size"]])

# Ordinal encoding - for ordinal categories
encoder = OrdinalEncoder(categories=[["small", "medium", "large"]])

# Target encoding - for high cardinality (uses target mean per category)
encoder = TargetEncoder(smoothing=10)
X["city_encoded"] = encoder.fit_transform(X["city"], y)

# Binary encoding - for high cardinality (fewer columns than one-hot)
encoder = BinaryEncoder(cols=["zip_code"])

# Hashing encoding - for very high cardinality (fixed output size)
encoder = HashingEncoder(n_components=8, cols=["url"])

# Frequency encoding - manual
freq = X["category"].value_counts(normalize=True)
X["category_freq"] = X["category"].map(freq)
```

### 3. Text Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# TF-IDF - lightweight, interpretable
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(texts)

# Sentence embeddings - semantic similarity, transfer learning
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Basic text features
df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["avg_word_length"] = df["text"].apply(
    lambda x: np.mean([len(w) for w in x.split()])
)
```

### 4. Time-Series Features

```python
def create_time_features(df, date_col="date", target_col="value"):
    """Create comprehensive time-series features."""
    df = df.sort_values(date_col).copy()

    # Lag features
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df[target_col].rolling(window).mean()
        df[f"rolling_std_{window}"] = df[target_col].rolling(window).std()
        df[f"rolling_min_{window}"] = df[target_col].rolling(window).min()
        df[f"rolling_max_{window}"] = df[target_col].rolling(window).max()

    # Expanding statistics
    df["expanding_mean"] = df[target_col].expanding().mean()

    # Date features
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    df["quarter"] = df[date_col].dt.quarter

    # Cyclical encoding for periodic features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df
```

### 5. Missing Value Imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Simple imputation
imputer = SimpleImputer(strategy="median")  # mean, median, most_frequent, constant

# KNN imputation - uses similar samples
imputer = KNNImputer(n_neighbors=5, weights="distance")

# MICE (Multiple Imputation by Chained Equations)
imputer = IterativeImputer(max_iter=10, random_state=42)

# Strategy by data type
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")
```

### 6. Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
import shap

# Filter method: Mutual Information
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Wrapper method: Recursive Feature Elimination
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(model, n_features_to_select=20, step=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

# Embedded method: L1 regularization
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]

# SHAP-based selection
model = RandomForestClassifier(n_estimators=100).fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
importance = np.abs(shap_values).mean(axis=0)
top_features = X.columns[np.argsort(importance)[-20:]]
```

### 7. Production Feature Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define feature groups
numeric_features = ["age", "income", "score"]
categorical_features = ["category", "region"]
text_features = ["description"]

# Build pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
    ("text", TfidfVectorizer(max_features=1000), "description"),
])

# Full pipeline with model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier()),
])

# Fit and save
pipeline.fit(X_train, y_train)
import joblib
joblib.dump(pipeline, "feature_pipeline.joblib")
```

### 8. Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from umap import UMAP

# PCA - linear, preserves variance
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# UMAP - non-linear, preserves local structure
reducer = UMAP(n_components=50, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)
```

## Best Practices

1. **Build sklearn Pipelines** - Ensure train/serve consistency
2. **Avoid data leakage** - Fit transformers on train data only
3. **Handle unknown categories** - Use `handle_unknown="ignore"` in encoders
4. **Version feature pipelines** - Save fitted transformers as artifacts
5. **Document feature semantics** - What each feature means and how it's computed
6. **Profile features** - Check distributions, correlations, missing rates
7. **Start simple** - Raw features + scaling before complex engineering
8. **Use target encoding carefully** - Always with cross-validation to avoid leakage
9. **Monitor feature distributions** in production for drift

## Scripts

- `scripts/transform_features.py` - Configurable feature transformation pipeline
- `scripts/select_features.py` - Multi-method feature selection

## References

See [references/REFERENCE.md](references/REFERENCE.md) for detailed comparisons.
