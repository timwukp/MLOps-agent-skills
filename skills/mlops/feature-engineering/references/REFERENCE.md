# Feature Engineering Reference Guide

## sklearn ColumnTransformer and Pipeline Patterns

### Basic Pipeline Structure

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ["age", "income", "credit_score"]
categorical_features = ["occupation", "city", "education"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
], remainder="drop")

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression()),
])
full_pipeline.fit(X_train, y_train)
```

---

## Feature Transformation Cheat Sheet

| Situation                        | Recommended Transform         | Notes                                         |
|----------------------------------|-------------------------------|-----------------------------------------------|
| Gaussian-like distribution       | `StandardScaler`              | Centers mean=0, std=1                         |
| Uniform or unknown distribution  | `MinMaxScaler`                | Scales to [0, 1]; sensitive to outliers       |
| Heavy outliers present           | `RobustScaler`                | Uses IQR; resistant to outliers               |
| Right-skewed positive values     | `np.log1p` / `PowerTransformer` | Reduces skew                               |
| Low-cardinality categories       | `OneHotEncoder`               | Creates binary columns per category           |
| High-cardinality categories      | `TargetEncoder` or hashing    | Avoids dimension explosion                    |
| Ordinal categories (e.g., size)  | `OrdinalEncoder`              | Preserves ordering                            |
| Tree-based models                | Minimal or no scaling needed  | Trees split on thresholds, not magnitudes     |
| Neural networks / SVM / KNN      | Always scale features         | Distance/gradient-based methods need scaling  |

---

## Handling Missing Data: Strategies Comparison

| Strategy                  | When to Use                              | Pros                          | Cons                                   |
|---------------------------|------------------------------------------|-------------------------------|----------------------------------------|
| **Drop rows**             | <5% missing, MCAR                        | Simple                        | Loses data; biased if not MCAR         |
| **Drop columns**          | >50% missing in a column                 | Removes noise                 | Loses potentially useful signal        |
| **Mean/Median imputation**| Numeric, low missing rate                | Fast, keeps sample size       | Distorts variance and correlations     |
| **Mode imputation**       | Categorical, low missing rate            | Simple                        | Bias toward frequent class             |
| **KNN imputation**        | Moderate missing, correlated features    | Leverages relationships       | Slow on large datasets                 |
| **Iterative (MICE)**      | Complex patterns, multiple columns       | Multivariate patterns         | Computationally expensive              |
| **Indicator column**      | Missingness itself is informative        | Preserves signal              | Adds extra dimension                   |
| **Forward/backward fill** | Time series data                         | Temporal structure            | Inappropriate for non-temporal data    |

---

## Feature Engineering by Data Type

### Numerical Features

- **Binning**: `KBinsDiscretizer` for converting continuous to ordinal.
- **Polynomial features**: `PolynomialFeatures(degree=2, interaction_only=True)` for interactions.
- **Ratio features**: `feature_a / feature_b` (e.g., debt-to-income ratio).
- **Aggregations**: Rolling means, group-level statistics (mean, std, count).

### Categorical Features

- **One-hot encoding**: Best for low cardinality (<15 categories).
- **Target encoding**: Mean of target per category, with regularization to avoid leakage.
- **Frequency encoding**: Replace category with its occurrence count.
- **Embedding layers**: For very high cardinality in deep learning models.

### Text Features

- **TF-IDF**: `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`.
- **Text statistics**: Length, word count -- simple but often predictive.
- **Sentence embeddings**: Pre-trained models (sentence-transformers) for semantic features.

### Datetime Features

```python
def extract_datetime_features(df, col):
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    # Cyclical encoding
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
    return df
```

### Geospatial Features

- **Haversine distance** to reference points (city center, nearest store).
- **Geohash encoding** for region bucketing.
- **Cluster membership** via KMeans on (latitude, longitude).

---

## Automated Feature Engineering Tools

### Featuretools (Deep Feature Synthesis)

```python
import featuretools as ft

es = ft.EntitySet(id="ecommerce")
es.add_dataframe(dataframe=customers_df, dataframe_name="customers", index="customer_id")
es.add_dataframe(dataframe=orders_df, dataframe_name="orders", index="order_id",
                 time_index="order_date")
es.add_relationship("customers", "customer_id", "orders", "customer_id")

feature_matrix, feature_defs = ft.dfs(
    entityset=es, target_dataframe_name="customers", max_depth=2,
    agg_primitives=["mean", "sum", "count", "std", "max", "min"],
    trans_primitives=["month", "weekday", "is_weekend"],
)
```

### tsfresh (Time Series Feature Extraction)

```python
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

features = extract_features(timeseries_df, column_id="sensor_id",
                            column_sort="timestamp", column_value="reading")
features = impute(features)
```

| Tool             | Best For                       | Scalability       |
|------------------|--------------------------------|-------------------|
| **Featuretools** | Relational / multi-table data  | Moderate (Dask)   |
| **tsfresh**      | Time series sensor data        | Moderate (parallel)|
| **AutoFeat**     | Automated polynomial features  | Small-medium data |
| **Feature Engine**| sklearn-compatible transforms | Good              |

---

## Feature Engineering Anti-Patterns

**1. Target Leakage**: Using features that encode future information. Fix: ask "would I know this before prediction?"

**2. Training/Serving Skew**: Computing features differently in training vs. serving. Fix: use serializable pipelines and a feature store.

**3. Encoding Before Splitting**: Fitting encoders on the full dataset before train/test split. Fix: always split first.

**4. Cardinality Explosion**: One-hot encoding 10,000 categories. Fix: use target encoding, frequency encoding, or embeddings.

**5. Over-Engineering**: Hundreds of polynomial features without selection. Fix: apply feature selection (mutual info, L1) after generation.

**6. Not Handling Unseen Categories**: Fix: use `handle_unknown="ignore"` or an "other" bucket.

---

## Best Practices

- Version your feature engineering pipeline alongside model code.
- Document each feature: its source, transformation logic, and business meaning.
- Monitor feature distributions in production for drift detection.
- Keep transformations deterministic and idempotent.
- Profile feature importance periodically and prune unused features.
- Prefer declarative pipelines (sklearn Pipeline) over imperative scripts.

---

## Further Reading

- [sklearn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Featuretools Documentation](https://featuretools.alteryx.com/en/stable/)
- [tsfresh Documentation](https://tsfresh.readthedocs.io/)
- [Feature Engineering and Selection (Kuhn & Johnson)](http://www.feat.engineering/)
- [Feature Engineering for ML (Zheng & Casari, O'Reilly)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
