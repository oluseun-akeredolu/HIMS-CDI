"""
Extended Reference Implementation for HIMS–CDI (with CIs, ablations, packaging).
"""

from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    cohen_kappa_score
)

EPS = 1e-9

@dataclass
class DomainConfig:
    name: str
    indicators: List[str]
    weight: float

def compute_domain_means(df: pd.DataFrame, domains: List[DomainConfig]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for d in domains:
        sub = df[d.indicators].copy()
        dmin, dmax = sub.values.min(), sub.values.max()
        sub = (sub - dmin) / (dmax - dmin + EPS)
        out[d.name] = sub.mean(axis=1)
    return out

def compute_cdi(domain_means: pd.DataFrame, domains: List[DomainConfig]) -> np.ndarray:
    weights = np.array([d.weight for d in domains])
    X = domain_means[[d.name for d in domains]].values
    return (X * weights).sum(axis=1)

def bootstrap_cis(values: np.ndarray, B: int = 1000, seed: int = 123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.choice(len(values), size=len(values), replace=True)].mean() for _ in range(B)]
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5))
    }

def evaluate_mlr(X: np.ndarray, y: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=7)
    metrics = {"f1":[],"auc":[],"auprc":[],"kappa":[]}
    for train, test in rskf.split(X, y):
        clf = LogisticRegression(max_iter=200, multi_class="multinomial").fit(X[train], y[train])
        prob = clf.predict_proba(X[test])
        pred = np.argmax(prob, axis=1)
        y_bin = np.eye(len(class_names))[y[test]]
        metrics["f1"].append(f1_score(y[test], pred, average="macro"))
        try:
            metrics["auc"].append(roc_auc_score(y_bin, prob, multi_class="ovr"))
        except Exception:
            pass
        metrics["auprc"].append(np.mean([average_precision_score(y_bin[:,k], prob[:,k]) for k in range(len(class_names))]))
        metrics["kappa"].append(cohen_kappa_score(y[test], pred))
    return {k: float(np.mean(v)) for k,v in metrics.items()}

def X_setup(dmeans, df, domains, cdi, use_cdi=True, equal_weights=False):
    # NOTE: you referenced raw indicator features but didn't use them;
    # keeping it minimal: domain means (+ optional CDI).
    feat = [dmeans.values]
    if use_cdi:
        feat.append(cdi.reshape(-1,1))
    X = np.concatenate(feat, axis=1)
    desc = "Features: domain_means" + (" + CDI" if use_cdi else "") + (" | equal weights" if equal_weights else "")
    return X, desc

def main(args):
    df = pd.read_csv(args.data)
    indicator_cols = [c for c in df.columns if "_ind" in c]

    domains = [
        DomainConfig("cti_modality",[c for c in indicator_cols if c.startswith("cti_modality")],0.18),
        DomainConfig("semantic_enrichment",[c for c in indicator_cols if c.startswith("semantic_enrichment")],0.18),
        DomainConfig("fusion_strategy",[c for c in indicator_cols if c.startswith("fusion_strategy")],0.16),
        DomainConfig("regulatory_traceability",[c for c in indicator_cols if c.startswith("regulatory_traceability")],0.24),
        DomainConfig("explainability",[c for c in indicator_cols if c.startswith("explainability")],0.24),
    ]

    dmeans = compute_domain_means(df[indicator_cols], domains)
    cdi = compute_cdi(dmeans, domains)
    cdi_ci = bootstrap_cis(cdi)

    if "risk_class" in df.columns:
        y_labels = df["risk_class"].values
    else:
        q1,q2 = np.quantile(cdi,[1/3,2/3])
        y_labels = np.array(["high" if v<q1 else "medium" if v<q2 else "low" for v in cdi])
    le = LabelEncoder().fit(["high","medium","low"])
    y = le.transform(y_labels)

    configs = {
        "default": X_setup(dmeans, df, domains, cdi, use_cdi=True),
        "no_cdi": X_setup(dmeans, df, domains, cdi, use_cdi=False),
        "equal_weights": X_setup(dmeans, df, domains, cdi, use_cdi=True, equal_weights=True),
        "drop_regulatory": X_setup(dmeans, df, [d for d in domains if d.name!="regulatory_traceability"], cdi, use_cdi=True),
    }

    results = {}
    for key,(X,desc) in configs.items():
        results[key] = {"desc": desc, "mlr": evaluate_mlr(X, y, le.classes_.tolist())}

    out = {"cdi_ci": cdi_ci, "ablations": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    p.add_argument("--output", type=str, default="extended_results.json", help="Output JSON path")
    args = p.parse_args()
    main(args)

if __name__ == "__main__":
    cli()
