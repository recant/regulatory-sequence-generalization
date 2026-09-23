import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

DNA = np.array(list("ACGT"))

def make_demo(seed=4, n_families=80, members_per_family=8, length=220):
    """Create related sequence families where row-level splitting leaks family identity."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in range(n_families):
        label = int(rng.integers(0, 2))
        founder = rng.choice(DNA, size=length)
        signature = rng.choice(DNA, size=10)
        founder[90:100] = signature
        for member in range(members_per_family):
            seq = founder.copy()
            mutate = rng.random(length) < 0.06
            seq[mutate] = rng.choice(DNA, size=mutate.sum())
            rows.append({"sequence": "".join(seq), "label": label, "family": f"fam_{fam:03d}"})
    return pd.DataFrame(rows)

def evaluate(df):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(6, 8), min_df=2, max_features=30000),
        LogisticRegression(max_iter=2000),
    )
    y = df["label"].astype(int)
    seq = df["sequence"].astype(str)

    random_cv = StratifiedKFold(5, shuffle=True, random_state=1)
    p_random = cross_val_predict(model, seq, y, cv=random_cv, method="predict_proba")[:, 1]

    group_cv = GroupKFold(5)
    p_group = cross_val_predict(model, seq, y, cv=group_cv, groups=df["family"], method="predict_proba")[:, 1]

    return pd.DataFrame([
        {"split": "random row split", "metric": "AUROC", "value": roc_auc_score(y, p_random)},
        {"split": "family-held-out split", "metric": "AUROC", "value": roc_auc_score(y, p_group)},
        {"split": "random row split", "metric": "AUPRC", "value": average_precision_score(y, p_random)},
        {"split": "family-held-out split", "metric": "AUPRC", "value": average_precision_score(y, p_group)},
    ])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    df = make_demo(seed=args.seed)
    result = evaluate(df)
    result.to_csv("example_results.csv", index=False)
    print(result.to_string(index=False))
