"""Accumulated Local Effects (ALE) on transparent surrogate models (numpy only).

The reports hold per-document results, not callable models, so each ALE here
is computed on a surrogate fitted to those results: ridge regression for a
continuous outcome, L2-regularized logistic regression for a binary one, with
standardized numeric features, their squares and pairwise products, and
categorical dummies (plus category × numeric interactions). ALE follows Apley &
Zhu (2020): quantile bins, per-bin mean finite differences, accumulated and
centered. Bands are 5th–95th percentiles over bootstrap refits; fit quality is
5-fold cross-validated (R² or AUC) so each curve states how far to trust it.
Outputs are rounded so rebuilds are byte-stable.

Vendored from Exios66/local-mailroom-sandbox ``reports/dashboard/ale.py`` (the
reports hub), with one change: bootstrap refits that diverge (a resample with
no errors has no finite logistic fit) are dropped rather than aborting the
curve; the band then covers the stable refits.
"""

from __future__ import annotations

import numpy as np

SEED = 20260928


class Design:
    def __init__(self, num: list[str], cat: str | None, rows: list[dict]):
        self.num, self.cat = num, cat
        X = np.array([[float(r[f]) for f in num] for r in rows])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.levels = sorted({r[cat] for r in rows}) if cat else []

    def matrix(self, X: np.ndarray, cats: list) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1])] + [Z[:, i] ** 2 for i in range(Z.shape[1])]
        cols += [Z[:, i] * Z[:, j] for i in range(Z.shape[1]) for j in range(i + 1, Z.shape[1])]
        for lv in self.levels[1:]:
            d = np.array([1.0 if c == lv else 0.0 for c in cats])
            cols.append(d)
            cols += [d * Z[:, i] for i in range(Z.shape[1])]
        return np.column_stack(cols)


def fit(M: np.ndarray, y: np.ndarray, kind: str, lam: float = 1.0) -> np.ndarray:
    P = np.eye(M.shape[1]) * lam
    P[0, 0] = 0
    if kind == "ridge":
        return np.linalg.solve(M.T @ M + P, M.T @ y)
    w = np.zeros(M.shape[1])
    for _ in range(100):  # IRLS / Newton with L2 penalty
        p = 1 / (1 + np.exp(-(M @ w)))
        g = M.T @ (p - y) + P @ w
        H = (M * (p * (1 - p))[:, None]).T @ M + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(M, w, kind):
    z = M @ w
    return z if kind == "ridge" else 1 / (1 + np.exp(-z))


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def cv_quality(D, X, cats, y, kind, rng):
    idx = rng.permutation(len(y))
    pred = np.zeros(len(y))
    for k in range(5):
        te = idx[k::5]
        tr = np.setdiff1d(idx, te)
        w = fit(D.matrix(X[tr], [cats[i] for i in tr]), y[tr], kind)
        pred[te] = predict(D.matrix(X[te], [cats[i] for i in te]), w, kind)
    if kind == "ridge":
        return {"metric": "R²", "value": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    return {"metric": "AUC", "value": auc(y, pred)}


def _ale_once(D, X, cats, w, kind, j, edges):
    xj = X[:, j]
    k_idx = np.clip(np.searchsorted(edges, xj, side="left") - 1, 0, len(edges) - 2)
    eff, cnt = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    lo, hi = X.copy(), X.copy()
    lo[:, j], hi[:, j] = edges[k_idx], edges[k_idx + 1]
    diff = predict(D.matrix(hi, cats), w, kind) - predict(D.matrix(lo, cats), w, kind)
    for k in range(len(edges) - 1):
        m = k_idx == k
        cnt[k] = m.sum()
        eff[k] = diff[m].mean() if m.any() else 0.0
    acc = np.concatenate([[0.0], np.cumsum(eff)])
    mid = (acc[:-1] + acc[1:]) / 2
    return acc - (mid * cnt).sum() / max(cnt.sum(), 1)


def ale(rows: list[dict], outcome: str, num: list[str], feature: str, cat: str | None = None,
        kind: str = "ridge", bins: int = 10, boot: int = 200, subset=None) -> dict:
    """ALE of ``feature`` on ``outcome``; ``subset`` restricts which rows are averaged (conditional ALE)."""
    rng = np.random.default_rng(SEED)
    D = Design(num, cat, rows)
    X = np.array([[float(r[f]) for f in num] for r in rows])
    cats = [r[cat] for r in rows] if cat else [None] * len(rows)
    y = np.array([float(r[outcome]) for r in rows])
    j = num.index(feature)
    sel = np.array([subset(r) for r in rows]) if subset else np.ones(len(rows), bool)
    xs = X[sel, j]
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, bins + 1)))
    if len(edges) < 5:  # heavily tied / discrete feature: one bin per observed value
        edges = np.unique(xs)
    if len(edges) < 3:
        raise ValueError(f"{feature}: too few distinct values for ALE")
    w = fit(D.matrix(X, cats), y, kind)
    Xs, cs = X[sel], [c for c, s in zip(cats, sel, strict=False) if s]
    main = _ale_once(D, Xs, cs, w, kind, j, edges)
    draws = []
    for _ in range(boot):
        b = rng.integers(0, len(y), len(y))
        try:
            wb = fit(D.matrix(X[b], [cats[i] for i in b]), y[b], kind)
        except np.linalg.LinAlgError:  # resample with no errors: logistic fit diverges
            continue
        if np.all(np.isfinite(wb)):
            draws.append(_ale_once(D, Xs, cs, wb, kind, j, edges))
    if len(draws) < boot // 2:
        raise ValueError(f"{feature}: too few stable bootstrap refits ({len(draws)}/{boot})")
    draws = np.array(draws)
    q = cv_quality(D, X, cats, y, kind, rng)
    r4 = lambda a: [round(float(v), 4) for v in a]  # noqa: E731
    return {
        "feature": feature, "outcome": outcome, "kind": kind, "n": int(sel.sum()), "n_fit": len(y),
        "x": r4(edges), "ale": r4(main), "lo": r4(np.percentile(draws, 5, axis=0)), "hi": r4(np.percentile(draws, 95, axis=0)),
        "rug": r4(np.quantile(xs, np.linspace(0.05, 0.95, 19))),
        "fit": {"metric": q["metric"], "value": None if q["value"] is None else round(q["value"], 3)},
        "range": round(float(main.max() - main.min()), 4),
    }
