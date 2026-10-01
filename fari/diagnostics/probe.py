from __future__ import annotations

import numpy as np


def hidden_state_separability(
    source_features,
    target_features,
    *,
    groups=None,
    cv_folds: int = 5,
    random_state: int = 0,
) -> dict:
    """Evaluate source/target linear separability with grouped cross-validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    source = np.asarray(source_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must be [N,D] with the same D")
    features = np.concatenate([source, target], axis=0)
    labels = np.concatenate([np.zeros(len(source), dtype=np.int64), np.ones(len(target), dtype=np.int64)])
    estimator = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=random_state),
    )
    if groups is not None:
        base_groups = np.asarray(groups)
        if len(base_groups) != len(source) or len(source) != len(target):
            raise ValueError("groups must contain one identifier per paired source/target sample")
        all_groups = np.concatenate([base_groups, base_groups])
        splitter = GroupKFold(n_splits=min(int(cv_folds), len(np.unique(base_groups))))
        splits = splitter.split(features, labels, all_groups)
    else:
        splitter = StratifiedKFold(n_splits=min(int(cv_folds), len(source)), shuffle=True, random_state=random_state)
        splits = splitter.split(features, labels)
    predictions = np.empty_like(labels)
    probabilities = np.empty(len(labels), dtype=np.float64)
    fold_count = 0
    for train, test in splits:
        model = estimator()
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
        fold_count += 1
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "fold_count": int(fold_count),
        "sample_count": int(len(labels)),
        "feature_dim": int(features.shape[1]),
    }
