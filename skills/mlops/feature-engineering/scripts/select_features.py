#!/usr/bin/env python3
"""Feature selection toolkit for ML pipelines.

Usage:
    python select_features.py --input data.csv --target label --method filter --n-features 20
    python select_features.py --input data.parquet --target y --method wrapper --n-features 15
    python select_features.py --input data.csv --target label --method embedded --n-features 10
    python select_features.py --input data.csv --target label --method all \
        --n-features 20 --output selected.json --report importance.png
"""
import argparse, json, logging, sys

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


def _prepare_Xy(df, target):
    if target not in df.columns:
        logger.error(f"Target '{target}' not found"); sys.exit(1)
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=["number"]).fillna(0)
    logger.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} numeric features")
    return X, y


def _is_clf(y):
    return y.dtype == "object" or y.nunique() < 20

# -- Filter methods ---------------------------------------------------------

def variance_threshold(X, threshold=0.01):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=threshold); sel.fit(X)
    kept = X.columns[sel.get_support()].tolist()
    logger.info(f"Variance filter kept {len(kept)}/{len(X.columns)} features")
    return kept


def correlation_filter(X, threshold=0.90):
    import numpy as np
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    logger.info(f"Correlation filter dropped {len(to_drop)} features (>{threshold})")
    return [c for c in X.columns if c not in to_drop]


def mutual_information_ranking(X, y, n):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    mi = (mutual_info_classif if _is_clf(y) else mutual_info_regression)(X, y, random_state=42)
    scores = dict(zip(X.columns, mi))
    ranked = sorted(scores, key=scores.get, reverse=True)[:n]
    logger.info(f"Mutual information top-{n} computed")
    return ranked, scores


def chi_squared_ranking(X, y, n):
    from sklearn.feature_selection import chi2
    try:
        chi_scores, _ = chi2(X.clip(lower=0), y)
    except Exception as e:
        logger.warning(f"Chi-squared failed: {e}"); return list(X.columns[:n]), {}
    scores = dict(zip(X.columns, chi_scores))
    return sorted(scores, key=scores.get, reverse=True)[:n], scores


def run_filter_methods(X, y, n):
    var_kept = variance_threshold(X)
    corr_kept = correlation_filter(X)
    mi_kept, mi_scores = mutual_information_ranking(X, y, n)
    chi_kept, chi_scores = chi_squared_ranking(X, y, n)
    structural = set(var_kept) & set(corr_kept)
    selected = [f for f in mi_kept if f in structural][:n]
    if len(selected) < n:
        selected.extend([f for f in mi_kept if f not in selected][:n - len(selected)])
    return {"variance_threshold": var_kept, "correlation_filter": corr_kept,
            "mutual_information": mi_kept,
            "mi_scores": {k: round(float(v), 6) for k, v in mi_scores.items()},
            "chi_squared": chi_kept,
            "chi_scores": {k: round(float(v), 4) for k, v in chi_scores.items()},
            "selected": selected[:n]}

# -- Wrapper methods --------------------------------------------------------

def recursive_feature_elimination(X, y, n):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import RFE
    est = (RandomForestClassifier if _is_clf(y) else RandomForestRegressor)(
        n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(est, n_features_to_select=n, step=1); rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    logger.info(f"RFE selected {len(selected)} features")
    return {"selected": selected, "rankings": dict(zip(X.columns, rfe.ranking_.tolist()))}

# -- Embedded methods -------------------------------------------------------

def lasso_selection(X, y, n):
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    X_sc = StandardScaler().fit_transform(X)
    if _is_clf(y):
        m = LogisticRegressionCV(penalty="l1", solver="saga", cv=5, max_iter=2000, random_state=42)
        m.fit(X_sc, y)
        imp = np.abs(m.coef_).mean(axis=0) if m.coef_.ndim > 1 else np.abs(m.coef_)
    else:
        m = LassoCV(cv=5, random_state=42, max_iter=2000); m.fit(X_sc, y)
        imp = np.abs(m.coef_)
    scores = dict(zip(X.columns, imp.tolist()))
    logger.info(f"Lasso selected top-{n} features")
    return sorted(scores, key=scores.get, reverse=True)[:n], scores


def tree_importance(X, y, n):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import numpy as np
    rf = (RandomForestClassifier if _is_clf(y) else RandomForestRegressor)(
        n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_sc = dict(zip(X.columns, rf.feature_importances_.tolist()))
    xgb_sc = {}
    try:
        import xgboost as xgb
        mdl = (xgb.XGBClassifier if _is_clf(y) else xgb.XGBRegressor)(
            n_estimators=200, random_state=42, verbosity=0)
        mdl.fit(X, y)
        xgb_sc = dict(zip(X.columns, mdl.feature_importances_.tolist()))
        logger.info("XGBoost importance computed")
    except ImportError:
        logger.info("XGBoost not installed; RandomForest only")
    combined = {c: float(np.mean([rf_sc[c]] + ([xgb_sc[c]] if c in xgb_sc else [])))
                for c in X.columns}
    selected = sorted(combined, key=combined.get, reverse=True)[:n]
    logger.info(f"Tree importance selected {len(selected)} features")
    return selected, {"rf": rf_sc, "xgb": xgb_sc, "combined": combined}


def boruta_selection(X, y):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import numpy as np, pandas as pd
    n_iter, hits = 20, {c: 0 for c in X.columns}
    for i in range(n_iter):
        shadow = X.apply(np.random.permutation).rename(columns=lambda c: f"shadow_{c}")
        merged = pd.concat([X, shadow], axis=1)
        rf = (RandomForestClassifier if _is_clf(y) else RandomForestRegressor)(
            n_estimators=100, random_state=i, n_jobs=-1, max_depth=7)
        rf.fit(merged, y)
        imp = rf.feature_importances_
        shadow_max = imp[len(X.columns):].max()
        for j, col in enumerate(X.columns):
            if imp[j] > shadow_max:
                hits[col] += 1
    selected = [c for c, h in hits.items() if h >= n_iter * 0.5]
    logger.info(f"Boruta selected {len(selected)}/{len(X.columns)} features")
    return selected, hits


def run_embedded_methods(X, y, n):
    lasso_f, lasso_s = lasso_selection(X, y, n)
    tree_f, tree_s = tree_importance(X, y, n)
    boruta_f, boruta_h = boruta_selection(X, y)
    return {"lasso": lasso_f, "lasso_scores": {k: round(v, 6) for k, v in lasso_s.items()},
            "tree_importance": tree_f,
            "tree_scores": {k: {fk: round(fv, 6) for fk, fv in v.items()}
                            for k, v in tree_s.items() if v},
            "boruta": boruta_f, "boruta_hits": boruta_h, "selected": tree_f[:n]}

# -- Visualization ----------------------------------------------------------

def plot_importance(scores, title, output_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]
    feats, vals = [x[0] for x in items][::-1], [x[1] for x in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.3)))
    ax.barh(feats, vals, color="#3b82f6")
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout(); fig.savefig(str(output_path), dpi=150); plt.close(fig)
    logger.info(f"Plot saved to {output_path}")

# -- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Feature selection for ML pipelines")
    ap.add_argument("--input", required=True, help="Input data file (CSV/Parquet/JSON)")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--method", default="all", choices=["filter", "wrapper", "embedded", "all"])
    ap.add_argument("--n-features", type=int, default=20, help="Features to select")
    ap.add_argument("--output", default=None, help="Output JSON path")
    ap.add_argument("--report", default=None, help="Output PNG for importance plot")
    args = ap.parse_args()

    logger.info(f"Loading {args.input}")
    df = load_data(args.input)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    X, y = _prepare_Xy(df, args.target)
    if X.shape[1] == 0:
        logger.error("No numeric features found."); sys.exit(1)

    n = min(args.n_features, X.shape[1])
    report = {"method": args.method, "n_features_requested": n, "results": {}}
    if args.method in ("filter", "all"):
        report["results"]["filter"] = run_filter_methods(X, y, n)
    if args.method in ("wrapper", "all"):
        report["results"]["wrapper"] = recursive_feature_elimination(X, y, n)
    if args.method in ("embedded", "all"):
        report["results"]["embedded"] = run_embedded_methods(X, y, n)

    if args.method == "all":
        from collections import Counter
        votes = Counter()
        for grp in report["results"].values():
            for f in grp.get("selected", []):
                votes[f] += 1
        report["consensus_selected"] = [f for f, _ in votes.most_common(n)]
        logger.info(f"Consensus: {report['consensus_selected']}")
    else:
        report["final_selected"] = report["results"].get(args.method, {}).get("selected", [])

    if args.report:
        scores = {}
        if "embedded" in report["results"]:
            scores = report["results"]["embedded"].get("tree_scores", {}).get("combined", {})
        if not scores and "filter" in report["results"]:
            scores = report["results"]["filter"].get("mi_scores", {})
        if scores:
            plot_importance(scores, "Feature Importance", args.report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
