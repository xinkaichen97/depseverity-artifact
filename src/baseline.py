"""Phase 2 - non-LLM reference points on the frozen split.

TF-IDF (word + char) -> SVD -> gradient boosting, in a plain multiclass form and
an ordinal (cumulative binary, Frank & Hall) form. Plus a majority-class floor.
This is a baseline, not a contribution.
"""
import json
import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import data as D
import evaluate as E

SEED = D.SEED


def featurize(train_txt, test_txt):
    """Returns (dense SVD features, raw sparse features, n_raw_features)."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                           sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=50_000, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    Xte = hstack([word.transform(test_txt), char.transform(test_txt)]).tocsr()
    # GBM needs dense; project rather than truncate the vocabulary.
    svd = TruncatedSVD(n_components=300, random_state=SEED)
    return (svd.fit_transform(Xtr), svd.transform(Xte)), (Xtr, Xte), Xtr.shape[1]


def gbm(balanced=False):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        class_weight="balanced" if balanced else None, random_state=SEED,
    )


def logreg(balanced=True):
    return LogisticRegression(C=4.0, max_iter=3000,
                              class_weight="balanced" if balanced else None)


def fit_multiclass(make, Xtr, ytr, Xte):
    return make().fit(Xtr, ytr).predict(Xte)


def fit_ordinal(make, Xtr, ytr, Xte):
    """Cumulative binary decomposition: P(y>0), P(y>1), P(y>2)."""
    cum = np.column_stack([make().fit(Xtr, (ytr > k).astype(int)).predict_proba(Xte)[:, 1]
                           for k in range(3)])
    # Enforce monotonicity, then convert to class probabilities.
    cum = np.minimum.accumulate(cum, axis=1)
    p = np.column_stack([1 - cum[:, 0], cum[:, 0] - cum[:, 1],
                         cum[:, 1] - cum[:, 2], cum[:, 2]])
    return p.argmax(1)


def main():
    df = D.load()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    (Dtr, Dte), (Str, Ste), nfeat = featurize(tr.text, te.text)
    ytr, yte = tr.y.values, te.y.values
    print(f"train {len(tr)}  test {len(te)}  tfidf features {nfeat} -> SVD 300\n")

    results = {"majority": E.majority_floor(yte)}
    for tag, make, X1, X2 in [
        ("gbm_svd", lambda: gbm(False), Dtr, Dte),
        ("gbm_svd_balanced", lambda: gbm(True), Dtr, Dte),
        ("logreg_sparse_balanced", lambda: logreg(True), Str, Ste),
    ]:
        results[f"{tag}_multiclass"] = E.evaluate(yte, fit_multiclass(make, X1, ytr, X2))
        results[f"{tag}_ordinal"] = E.evaluate(yte, fit_ordinal(make, X1, ytr, X2))

    for k, m in results.items():
        print(E.fmt(k, m))
    print()
    for k, m in results.items():
        print(f"{k} confusion (rows=gold {E.LABELS}):")
        for lab, row in zip(E.LABELS, m["confusion"]):
            print(f"  {lab:<9} {row}")
    with open("results/phase2_baseline.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
