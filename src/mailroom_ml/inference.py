"""Hierarchical ModernBERT intake inference (plan §4/§9, issues #85 M3/#88).

The classifier fast path: one shared encoder, six conditional heads
(doc_type + per-class subclass), window plurality-vote merge (the sorter's
merge), calibrated probabilities, the composite route score S = p·a·m
(plan D4), and the fast-path gate.  Fail-open is structural: every ML
dependency is wrapped and any exception becomes a structured failure the
routing layer turns into the LLM path (plan D10) — this module never
silently assigns a label.

Model loading (``load_bundle``): the artifact directory holds the label
maps (``labels.json`` — ALWAYS from the bundle, never hard-coded; the
committed ``labels.py`` canon is the guard reference only), the tokenizer
(``tokenizer.json``), optionally per-head temperatures (``temperatures.json``
— plan §8) and a session: ONNX ``model.onnx`` (fp32) preferred — the int8
``model_quantized.onnx`` variant flips argmax on the doc_type/contract/
corporate_record heads under dynamic quantization (parity gate 2026-09-21:
argmax agreement 0.8 / 0.2 / 0.6 over the gate samples) so fp32 serves by
default and int8 is a fallback only — or a PyTorch checkpoint
(``model.safetensors`` + ``heads.pt``).  Resolution order: env override
(``ML_MODEL_DIR``) > caller override > ``artifacts/onnx/model`` >
``artifacts/pytorch/model`` (ONNX bundle preferred when both exist).

Tests inject a stub bundle (``predict_fn`` + small maps) — no network, no
model downloads, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mailroom_ml.config import (
    ABSTAIN_UNKNOWN_CLASS,
    ARTIFACTS_DIR,
    BERT_INTAKE_MAX_CHARS,
    CATCH_ALL_LABELS,
    GATE_REQUIRED_AGREEMENT,
    MAX_TOKENS,
    ROUTE_DOC_CONFIDENCE,
    ROUTE_MARGIN,
    ROUTE_MIN_AUTHENTIC_SUPPORT,
    ROUTE_SUBCLASS_CONFIDENCE,
    ROUTE_WINDOW_AGREEMENT,
    SUBCLASS_PROJECTIONS,
    SUBCLASS_UNMAPPED_ROUTE,
    WINDOW_OVERLAP_TOKENS,
)
from mailroom_ml.labels import normalize_label_maps

__all__ = [
    "ModelBundle",
    "BundleUnavailable",
    "BundleLoadError",
    "resolve_model_dir",
    "load_bundle",
    "predict",
    "encode_inputs",
    "window_titles",
    "classify_windows",
    "classify_document",
    "project_subclass",
    "TriageResult",
]


class BundleUnavailable(RuntimeError):
    """No artifact bundle found (model resolution exhausted) — route LLM."""


class BundleLoadError(RuntimeError):
    """The resolved artifact directory is present but broken — route LLM."""


@dataclass
class ModelBundle:
    """Loaded artifact bundle: maps, tokenizer, session + provenance.

    ``predict_fn`` is the inference seam: tests inject a stub callable with
    the same signature (``def predict(input_ids, attention_mask) -> dict[str,
    np.ndarray]`` returning per-head logits).  The production loaders bind it
    to the ONNX session or the PyTorch model.
    """

    model_dir: Path
    maps: dict[str, Any]
    temperatures: dict[str, float] = field(default_factory=dict)
    support_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    tokenizer: Any | None = None
    pad_id: int | None = None
    artifact_sha: str | None = None
    model_kind: str = "unknown"  # "onnx-int8" | "onnx-fp32" | "pytorch" | "stub"
    predict_fn: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]] | None = None
    # #107 head-exclusion policy (from summary.json checkpoint_selection):
    # head name -> reason.  A subclass head whose calibrated ECE never
    # cleared the selection budget is excluded from the fast path — its
    # doc_type routes to the LLM instead.  Absent (pre-#107 artifacts) ->
    # no exclusions; the doc_type selection gate still applies.
    head_exclusions: dict[str, str] = field(default_factory=dict)
    exclusion_policy: dict[str, Any] | None = None

    @property
    def label_schema_version(self) -> str:
        """Surface identity for spans: head sizes + bundle revision hash."""
        sizes = {k: len(v.get("labels", [])) for k, v in self.maps.items()}
        return f"heads={json.dumps(sizes, sort_keys=True)}"


# ---------------------------------------------------------------------------
# Model resolution + loading
# ---------------------------------------------------------------------------

def _default_model_dir() -> Path | None:
    for cand in (ARTIFACTS_DIR / "onnx" / "model",
                 ARTIFACTS_DIR / "pytorch" / "model"):
        if (cand / "labels.json").is_file():
            return cand
    return None


def resolve_model_dir(override: str | Path | None = None) -> Path | None:
    """Order: env ``ML_MODEL_DIR`` > caller override > artifact defaults."""
    env = os.environ.get("ML_MODEL_DIR")
    if env:
        p = Path(env)
        return p if (p / "labels.json").is_file() else None
    if override:
        p = Path(override)
        return p if (p / "labels.json").is_file() else None
    return _default_model_dir()


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_support_counts(model_dir: Path) -> dict[str, dict[str, int]]:
    """Per-head authentic train counts (``train_counts.json`` sidecar).

    The ROUTE_MIN_AUTHENTIC_SUPPORT gate (fast path requires >= 5 real
    training rows per label) reads this sidecar; absent -> empty map -> the
    gate fails open to the LLM path (support unknown is not a fast path).
    """
    p = model_dir / "train_counts.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: {str(c): int(n) for c, n in v.items()}
                for k, v in data.items() if isinstance(v, dict)}
    except (ValueError, TypeError, OSError):
        return {}


def _load_head_exclusions(model_dir: Path) -> tuple[dict[str, str], dict | None]:
    """#107 head-exclusion policy from the artifact's ``summary.json``.

    Reads ``checkpoint_selection.head_exclusion_policy`` (written by the
    trainer's ``_summary``): heads whose calibrated ECE exceeded the budget
    are excluded from the fast path.  Absent or malformed -> no exclusions
    (pre-#107 artifacts); the doc_type selection gate still applies.
    """
    p = model_dir / "summary.json"
    if not p.is_file():
        return {}, None
    try:
        sel = json.loads(p.read_text(encoding="utf-8")).get(
            "checkpoint_selection", {})
    except (ValueError, OSError):
        return {}, None
    policy = sel.get("head_exclusion_policy")
    if not isinstance(policy, dict):
        return {}, None
    excluded = policy.get("excluded")
    if not isinstance(excluded, dict):
        return {}, policy
    reasons = {name: f"ece_calibrated {info.get('ece_calibrated')} > "
                     f"budget {policy.get('budget')}"
               for name, info in excluded.items()
               if isinstance(info, dict) and info.get("excluded")}
    return reasons, policy


def _load_tokenizer(model_dir: Path):
    """Standalone ``tokenizers`` Tokenizer from the bundle (no transformers).

    The trainer's tokenizer.json bakes in ``padding: longest`` +
    ``truncation: 8192`` (batch-training convenience).  Both are neutralized
    here: padding would silently stretch every window to the full 8,192-token
    tensor (minutes per CPU doc), and truncation would violate the
    no-truncation doctrine (``encode_inputs`` must see the TRUE length so an
    overflow routes to the LLM instead of being silently cut).
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def _resolve_pad_id(model_dir: Path, tok) -> int | None:
    for name in ("tokenizer_config.json", "config.json"):
        p = model_dir / name
        if not p.is_file():
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        pad = cfg.get("pad_token") or cfg.get("pad_token_id")
        if pad is None:
            continue
        if isinstance(pad, int):
            return pad
        pid = tok.token_to_id(pad)
        if pid is not None:
            return pid
    return None


def _onnx_session(model_dir: Path) -> tuple[str, Any]:
    import onnxruntime as ort

    quant = model_dir / "model_quantized.onnx"
    fp32 = model_dir / "model.onnx"
    # fp32 first: dynamic int8 quantization flips argmax on the doc_type /
    # contract / corporate_record heads (parity gate 2026-09-21); int8 is a
    # fallback only so a bundle without the fp32 graph still resolves.
    if fp32.is_file():
        return ("onnx-fp32", ort.InferenceSession(
            str(fp32), providers=["CPUExecutionProvider"]))
    if quant.is_file():
        return ("onnx-int8", ort.InferenceSession(
            str(quant), providers=["CPUExecutionProvider"]))
    raise BundleLoadError(f"no model.onnx / model_quantized.onnx under {model_dir}")


def _head_from_state(state: dict, hidden: int):
    """Reconstruct a head module from its state dict (linear vs MLP).

    Single-``nn.Linear`` heads save ``weight``/``bias``; ``--mlp-heads``
    heads save ``0.weight``/``0.bias``/``3.weight``/``3.bias`` (Sequential:
    Linear, SiLU, Dropout, Linear — train_modernbert.HierarchicalClassifier;
    the trained head kind is NOT recorded in the bundle, so the state dict is
    self-describing).  Dropout has no parameters; eval-mode it is identity,
    so the reconstruction uses 0.0.
    """
    import torch

    if any(k.startswith("0.") for k in state):
        return torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden, state["3.weight"].shape[0]),
        )
    return torch.nn.Linear(hidden, state["weight"].shape[0], bias=True)


def _pytorch_predict(model: Any, head_modules: dict[str, Any], device: Any):
    """Bind a PyTorch checkpoint to the predict_fn contract.

    ``model`` is the backbone (``AutoModel``), ``head_modules`` the
    reconstructed per-head modules (built by ``load_bundle`` via
    ``_head_from_state`` from ``heads.pt`` — linear or MLP, self-describing);
    the forward replicates the trainer: first-token pooling (ModernBERT has
    no pooler) + per-head logits.
    """
    import torch

    def predict(input_ids: np.ndarray, attention_mask: np.ndarray):
        ids = torch.as_tensor(input_ids, device=device)
        mask = torch.as_tensor(attention_mask, device=device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
            pooled = out.last_hidden_state[:, 0].float()
            return {name: head(pooled).cpu().numpy()
                    for name, head in head_modules.items()}

    return predict


def load_bundle(model_dir: str | Path | None = None,
                *, prefer_onnx: bool = True) -> ModelBundle:
    """Load one artifact bundle; raises BundleUnavailable/BundleLoadError."""
    resolved = resolve_model_dir(model_dir)
    if resolved is None:
        raise BundleUnavailable(
            "no classifier artifact bundle resolved (ML_MODEL_DIR or "
            "artifacts/{pytorch,onnx}/model with labels.json) — route LLM "
            "(fail-open)")
    mdir = Path(resolved)
    labels = mdir / "labels.json"
    if not labels.is_file():
        raise BundleLoadError(f"bundle {mdir} missing labels.json")

    maps = normalize_label_maps(json.loads(labels.read_text(encoding="utf-8")))
    try:
        tok = _load_tokenizer(mdir)
        pad_id = _resolve_pad_id(mdir, tok) or 0
    except Exception as exc:  # noqa: BLE001 — tokenizer load failures
        raise BundleLoadError(f"tokenizer load failed: {exc}") from exc
    temps: dict[str, float] = {}
    tp = mdir / "temperatures.json"
    if tp.is_file():
        try:
            temps = {k: float(v) for k, v in json.loads(tp.read_text()).items()}
        except (ValueError, TypeError, OSError):
            temps = {}

    kind = ""
    artifact_sha: str | None = None
    predict_fn: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]] | None = None

    if prefer_onnx and (mdir / "model.onnx").is_file() \
            or (mdir / "model_quantized.onnx").is_file():
        kind = "onnx"
        try:
            kind, sess = _onnx_session(mdir)
        except BundleLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 — onnxruntime load failures
            raise BundleLoadError(f"onnx session load failed: {exc}") from exc

        def _onnx_predict(input_ids: np.ndarray, attention_mask: np.ndarray):
            outs = sess.run(None, {"input_ids": input_ids,
                                   "attention_mask": attention_mask})
            names = [o.name for o in sess.get_outputs()]
            return {n.removeprefix("logits_"): a for n, a in zip(names, outs, strict=True)}

        predict_fn = _onnx_predict
        modelfile = mdir / ("model_quantized.onnx" if kind == "onnx-int8"
                            else "model.onnx")
        artifact_sha = _sha256(modelfile)
    elif (mdir / "model.safetensors").is_file() and (mdir / "heads.pt").is_file():
        kind = "pytorch"
        import torch
        from transformers import AutoModel

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Trainer semantics: bf16+sdpa on CUDA, fp32 on CPU; heads stay
            # fp32 (the trainer casts pooled -> float before the heads) —
            # matches the deployed predictor.
            model = AutoModel.from_pretrained(
                str(mdir),
                torch_dtype=torch.bfloat16 if device.type == "cuda"
                else torch.float32,
                attn_implementation="sdpa" if device.type == "cuda" else None,
            ).to(device)
            model.eval()
            heads = torch.load(mdir / "heads.pt", map_location=device)
            head_modules: dict[str, Any] = {}
            for name, state in heads.items():
                lin = _head_from_state(state, model.config.hidden_size).to(device)
                lin.load_state_dict(state)
                head_modules[name] = lin
        except Exception as exc:  # noqa: BLE001 — torch/transformers load errors
            raise BundleLoadError(f"pytorch checkpoint load failed: {exc}") from exc
        predict_fn = _pytorch_predict(model, head_modules, device)
        artifact_sha = _sha256(mdir / "model.safetensors")
    else:
        raise BundleLoadError(
            f"bundle {mdir} has neither ONNX graph nor pytorch checkpoint")

    head_exclusions, exclusion_policy = _load_head_exclusions(mdir)
    return ModelBundle(
        model_dir=mdir, maps=maps, temperatures=temps,
        tokenizer=tok, pad_id=pad_id,
        artifact_sha=artifact_sha, model_kind=kind, predict_fn=predict_fn,
        support_counts=_load_support_counts(mdir),
        head_exclusions=head_exclusions,
        exclusion_policy=exclusion_policy,
    )


# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------

def encode_inputs(bundle: ModelBundle, texts: list[str],
                  max_length: int = MAX_TOKENS) -> tuple[np.ndarray, np.ndarray]:
    """Encode without truncation; raises ValueError when a text overflows.

    No-truncation doctrine (epic #85/#88): BERT covers a document only when
    it fits the native context; an overflow here must NOT be silently
    truncated — the caller routes to the LLM path instead (coverage < 1.0).
    """
    encs = bundle.tokenizer.encode_batch(texts)
    n = len(texts)
    max_len = max(len(e.ids) for e in encs) or 1
    if max_len > max_length:
        raise ValueError(
            f"input of {max_len} tokens exceeds the {max_length}-token "
            f"ModernBERT context — route LLM (no truncation, #85)")
    ids = np.full((n, max_len), bundle.pad_id, dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.int64)
    for i, enc in enumerate(encs):
        ids[i, :len(enc.ids)] = enc.ids
        mask[i, :len(enc.ids)] = 1
    return ids, mask


def predict(bundle: ModelBundle,
            input_ids: np.ndarray, attention_mask: np.ndarray,
            ) -> dict[str, np.ndarray]:
    """Per-head logits; raises BundleLoadError when no predict_fn is bound."""
    if bundle.predict_fn is None:
        raise BundleLoadError("bundle has no predict_fn bound (stub missing)")
    return bundle.predict_fn(input_ids, attention_mask)


def window_titles(title: str, window_texts: list[str]) -> list[str]:
    """v1 input format: title + "\\n\\n" + window (plan D5, byte-compatible)."""
    return [f"{title}\n\n{w}" if title else w for w in window_texts]


def _calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    from mailroom_ml.calibration import apply_temperature

    return apply_temperature(logits, temperature)


def _trainable_logit_width(head_cfg: dict[str, Any]) -> int:
    """Number of logits the head may argmax over (trainable decision space)."""
    trainable = head_cfg.get("trainable_labels")
    if trainable:
        return len(trainable)
    if "labels" in head_cfg:
        return len(head_cfg["labels"])
    return len(head_cfg.get("label2id", {}))


def _argmax_trainable(probs: np.ndarray, head_cfg: dict[str, Any]) -> int:
    """Argmax restricted to the trainable label slice (#116).

    Legacy checkpoints whose ``heads.pt`` still emit one logit per full
    ``labels`` entry keep working: when ``probs`` is wider than the
    trainable slice, only the trainable prefix is scored.
    """
    n = _trainable_logit_width(head_cfg)
    row = probs if probs.ndim == 1 else probs
    if row.shape[-1] > n:
        row = row[..., :n]
    return int(row.argmax())


def _id2label_for_logits(head_cfg: dict[str, Any], n_logits: int) -> dict[str, str]:
    trainable = head_cfg.get("trainable_labels") or []
    if trainable and len(trainable) == n_logits:
        return head_cfg["trainable_id2label"]
    return head_cfg["id2label"]


def classify_windows(bundle: ModelBundle, window_texts: list[str],
                     temperatures: dict[str, float] | None = None,
                     max_length: int = MAX_TOKENS) -> dict[str, Any]:
    """Per-window hierarchical inference + the sorter's plurality merge.

    Every window runs the shared encoder; doc_type + the winning class's
    subclass head produce per-window votes.  The document vote mirrors the
    sorter merge (trainer ``evaluate``): doc_type by plurality over windows;
    subclass by plurality over windows whose doc_type vote is the winning
    class — windows voting ``unknown`` abstain from subclass (routed to the
    LLM by the gate, never a crash, plan §51/#43).

    Returns the composite route score S = p_calibrated × agreement × margin
    (plan D4) with agreement/margin/runner-up and per-head calibration.
    """
    if not window_texts:
        raise ValueError("classify_windows needs >= 1 window")

    # calibrated probs per head across all windows (batch once)
    ids, mask = encode_inputs(bundle, window_texts, max_length=max_length)
    logits_by_head = predict(bundle, ids, mask)
    temps = dict(bundle.temperatures) if temperatures is None else temperatures

    heads = sorted(k for k in logits_by_head if k != "doc_type")
    unknown = bundle.maps["doc_type"]["label2id"].get(ABSTAIN_UNKNOWN_CLASS)
    dt_head = bundle.maps["doc_type"]
    doc_map = _id2label_for_logits(dt_head, _trainable_logit_width(dt_head))

    # calibrated probabilities per head across all windows, batched once
    probs_by_head = {name: _calibrated_probs(
        logits_by_head[name], temps.get(name, 1.0))
        for name in logits_by_head}

    win_dt: list[int] = []          # per-window doc_type argmax (id space)
    win_sc: dict[str, list[int | None]] = defaultdict(list)
    doc_probs: list[np.ndarray] = []
    sub_probs: dict[str, list[np.ndarray]] = defaultdict(list)

    for i in range(len(window_texts)):
        p_dt = probs_by_head["doc_type"][i]
        doc_probs.append(p_dt)
        dt_id = _argmax_trainable(p_dt, bundle.maps["doc_type"])
        win_dt.append(dt_id)
        cls = doc_map[str(dt_id)]
        if cls in probs_by_head and dt_id != unknown:
            p_sc = probs_by_head[cls][i]
            sub_probs[cls].append(p_sc)
            win_sc[cls].append(_argmax_trainable(p_sc, bundle.maps[cls]))
        else:
            win_sc[cls].append(None)  # abstain — subclass not scored

    # doc_type plurality + agreement/margin/runner-up (calibrated)
    if unknown is not None:
        known_votes = [v for v in win_dt if v != unknown]
    else:
        known_votes = win_dt
    if not known_votes:
        dt_id = unknown if unknown is not None else int(win_dt[0])
        dt_label = doc_map[str(dt_id)]
        return {
            "doc_type": dt_label, "subclass": None, "route": "llm",
            "reason": "all_windows_abstained",
            "agreement": 0.0, "margin": 0.0, "runner_up": None,
            "score": 0.0, "n_windows": len(window_texts),
            "per_head": {}, "guard_failures": ["all_windows_unknown"],
            "window_doc_type_probs": doc_probs,
        }
    dt_id = Counter(known_votes).most_common(1)[0][0]
    dt_label = doc_map[str(dt_id)]
    n_agree = sum(1 for v in win_dt if v == dt_id)
    agreement = n_agree / len(window_texts)

    # calibrated confidences for the winning class per window
    mean_p = np.mean([p[dt_id] for p in doc_probs])
    # runner-up: second-highest mean calibrated probability
    mean_by_class = np.mean(np.stack(doc_probs), axis=0)
    order = np.argsort(mean_by_class)[::-1]
    runner_id: int | None = None
    for idx in order:
        if int(idx) != dt_id:
            runner_id = int(idx)
            break
    runner_label = doc_map[str(runner_id)] if runner_id is not None else None
    margin = float(mean_p - (mean_by_class[runner_id] if runner_id is not None else 0.0))

    # subclass plurality over windows voting the winning class (abstentions skip)
    cls = dt_label
    sc_pred: int | None = None
    sc_conf = 0.0
    n_scored = 0
    if cls in heads and dt_id != unknown:
        cond = [v for v in win_sc[cls] if v is not None]
        if cond:
            sc_pred = Counter(cond).most_common(1)[0][0]
            sc_probs = sub_probs[cls]
            sc_conf = float(np.mean([p[sc_pred] for p in sc_probs]))
            n_scored = len(sc_probs)
    sc_label = None
    if sc_pred is not None:
        id2label = _id2label_for_logits(
            bundle.maps[cls], _trainable_logit_width(bundle.maps[cls]))
        sc_label = id2label[str(sc_pred)]

    score = float(np.clip(mean_p, 0, 1)) * agreement * max(0.0, margin)
    per_head: dict[str, dict[str, Any]] = {}
    for h in heads:
        head_cfg = bundle.maps[h]
        n_train = _trainable_logit_width(head_cfg)
        id2label = _id2label_for_logits(head_cfg, n_train)
        preds = [
            id2label[str(_argmax_trainable(p, head_cfg))]
            for p in probs_by_head[h]]
        per_head[h] = {"pred": Counter(preds).most_common(1)[0][0],
                       "windows": preds}
    return {
        "doc_type": dt_label,
        "subclass": sc_label,
        "route": None,  # set by classify_document / routing gate
        "reason": None,
        "agreement": float(round(agreement, 4)),
        "margin": float(round(margin, 4)),
        "runner_up": runner_label,
        "score": float(round(score, 4)),
        "n_windows": len(window_texts),
        "n_class_windows": n_scored,
        "calibrated_confidence": float(round(float(mean_p), 4)),
        "subclass_confidence": float(round(sc_conf, 4)),
        "per_head": per_head,
        "guard_failures": [],
        "window_doc_type_probs": doc_probs,
    }


def project_subclass(maps: dict[str, Any], doc_type: str,
                     subclass: str | None) -> str | None:
    """Canonical -> head-vocab subclass projection (#107 taxonomy conformance).

    The trained heads express a subset of the canonical taxonomy (observed
    GT surfaces, #66/#67/#68).  A canonical subclass the head cannot express
    must project deterministically — never force-fit into a sibling class:

    - in-vocab label -> returned unchanged;
    - explicit projection rule (``SUBCLASS_PROJECTIONS``, e.g.
      ``certificate_of_formation -> other``) -> the projected label when the
      head carries it;
    - else a head with an ``other`` class -> ``other``;
    - else -> ``None`` (UNMAPPED — the caller routes to the LLM per
      ``SUBCLASS_UNMAPPED_ROUTE``; the insurance head has no ``other``, so
      any out-of-vocab insurance subclass is unmapped, never re-mapped).
    """
    if subclass is None:
        return None
    head = maps.get(doc_type)
    if head is None:
        return None
    labels = head.get("labels", [])
    if subclass in labels:
        return subclass
    projected = SUBCLASS_PROJECTIONS.get(doc_type, {}).get(subclass)
    if projected is not None and projected in labels:
        return projected
    if "other" in labels:
        return "other"
    return None


_DEFAULT_BUNDLE_CACHE: ModelBundle | None = None


def clear_default_bundle_cache() -> None:
    """Drop the cached default bundle (tests, hot reload, artifact swaps)."""
    global _DEFAULT_BUNDLE_CACHE
    _DEFAULT_BUNDLE_CACHE = None


def classify_document_default(
    doc_text: str,
    *,
    filename: str | None = None,
    max_chars: int = BERT_INTAKE_MAX_CHARS,
    max_tokens: int = MAX_TOKENS,
    overlap: int = WINDOW_OVERLAP_TOKENS,
    window_texts: list[str] | None = None,
    min_authentic_support: int = ROUTE_MIN_AUTHENTIC_SUPPORT,
    doc_confidence: float = ROUTE_DOC_CONFIDENCE,
    subclass_confidence: float = ROUTE_SUBCLASS_CONFIDENCE,
    window_agreement: float = ROUTE_WINDOW_AGREEMENT,
    margin_gate: float = ROUTE_MARGIN,
    agreement_gate: float = GATE_REQUIRED_AGREEMENT,
) -> dict[str, Any]:
    """Default-bundle convenience entrypoint (M6a lane seam, #102/#103).

    The graph lane (``agents/bert_intake.py``) calls ``classify_document(
    doc_text, filename=...)`` — that signature never existed; the real one
    requires a loaded ``ModelBundle`` + title. This seam is the single
    adapter: it resolves the DEFAULT bundle (env ``ML_MODEL_DIR`` > caller
    override > artifact defaults, cached per process), derives the title
    from ``filename``'s stem, and forwards to :func:`classify_document`.

    Fail-open by construction, same contract as the full entrypoint: a
    missing/unloadable bundle NEVER raises — the result carries a failure
    (``route="llm"``) with the machine-readable reason the lane's
    missing-model markers understand (``no_model`` / ``bundle_missing`` /
    ``model_missing``). Any classifier exception is likewise wrapped inside
    ``classify_document`` itself.
    """
    global _DEFAULT_BUNDLE_CACHE
    if _DEFAULT_BUNDLE_CACHE is None:
        try:
            _DEFAULT_BUNDLE_CACHE = load_bundle(None, prefer_onnx=True)
        except BundleUnavailable as exc:
            return {
                "status": "failure",
                "route": "llm",
                "reason": "no_model",
                "doc_type": None,
                "subclass": None,
                "detail": str(exc)[:200],
                "quality": {"coverage": 0.0, "messy": False},
                "guard_failures": [],
            }
        except BundleLoadError as exc:
            return {
                "status": "failure",
                "route": "llm",
                "reason": "bundle_missing" if "labels.json" in str(exc) else "model_missing",
                "doc_type": None,
                "subclass": None,
                "detail": str(exc)[:200],
                "quality": {"coverage": 0.0, "messy": False},
                "guard_failures": [],
            }
    title = Path(filename).stem if filename else "untitled"
    return classify_document(
        _DEFAULT_BUNDLE_CACHE,
        title,
        doc_text,
        max_chars=max_chars,
        max_tokens=max_tokens,
        overlap=overlap,
        window_texts=window_texts,
        min_authentic_support=min_authentic_support,
        doc_confidence=doc_confidence,
        subclass_confidence=subclass_confidence,
        window_agreement=window_agreement,
        margin_gate=margin_gate,
        agreement_gate=agreement_gate,
    )


def classify_document(bundle: ModelBundle, title: str, doc_text: str, *,
                      max_chars: int = BERT_INTAKE_MAX_CHARS,
                      max_tokens: int = MAX_TOKENS,
                      overlap: int = WINDOW_OVERLAP_TOKENS,
                      window_texts: list[str] | None = None,
                      min_authentic_support: int = ROUTE_MIN_AUTHENTIC_SUPPORT,
                      doc_confidence: float = ROUTE_DOC_CONFIDENCE,
                      subclass_confidence: float = ROUTE_SUBCLASS_CONFIDENCE,
                      window_agreement: float = ROUTE_WINDOW_AGREEMENT,
                      margin_gate: float = ROUTE_MARGIN,
                      agreement_gate: float = GATE_REQUIRED_AGREEMENT,
                      ) -> dict[str, Any]:
    """Full fast-path inference on one document (windows + merge + gate).

    Steps (plan §4): context-fit gate (chars/tokens — oversize routes LLM,
    never truncate), windowing (title + slide, 512 overlap; tests may hand
    prebuilt ``window_texts`` to skip the transformers windower), merge,
    composite score, gate.  Any ML exception is wrapped into the result as a
    failure (route "llm") — fail-open, never a raise to the caller.

    The fast path requires ALL of: supported doc_type (not ``unknown``),
    calibrated doc_type confidence >= ``ROUTE_DOC_CONFIDENCE``,
    subclass confidence >= ``ROUTE_SUBCLASS_CONFIDENCE`` when a subclass
    label is predicted (``subclass`` non-null), window agreement >=
    ``ROUTE_WINDOW_AGREEMENT``, margin >= ``ROUTE_MARGIN``, authentic support
    >= ``ROUTE_MIN_AUTHENTIC_SUPPORT``, no OOD flag, no guard failures.
    """
    import traceback as _tb

    result: dict[str, Any] = {
        "status": "ok", "doc_type": None, "subclass": None, "route": "llm",
        "reason": None, "score": 0.0, "quality": {}, "guard_failures": [],
        "failure": None, "artifact_sha": bundle.artifact_sha,
    }
    try:
        chars = len(doc_text)
        context_fit = chars <= max_chars
        result["quality"]["chars"] = chars
        result["quality"]["context_fit"] = context_fit
        if not context_fit:
            result["reason"] = "oversize_chars"
            result["guard_failures"].append("oversize_chars")
            result["quality"]["coverage"] = 0.0
            return result

        if window_texts is None:
            from mailroom_ml.windows import window_document

            window_texts = window_document(title, doc_text, max_tokens, overlap)
        result["quality"]["windows"] = len(window_texts)
        if not window_texts:
            result["reason"] = "no_windows"
            result["guard_failures"].append("no_windows")
            return result

        decorated = window_titles(title, window_texts)
        merged = classify_windows(bundle, decorated, max_length=max_tokens)
        result.update(merged)
        if merged.get("route") == "llm":  # all windows abstained
            return result

        dt = merged["doc_type"]
        sub = merged["subclass"]
        vocab_ok = dt in bundle.maps or dt == ABSTAIN_UNKNOWN_CLASS
        if not vocab_ok:
            result["reason"] = "vocab_fail"
            result["route"] = "llm"
            result["guard_failures"].append(
                f"doc_type {dt!r} not a supported head")
            return result

        # #107 head-exclusion policy: a subclass head whose calibrated ECE
        # never cleared the selection budget is excluded from the fast path
        # — its doc_type routes to the LLM (the subclass prediction cannot
        # be trusted for the conditional route).
        if dt in bundle.head_exclusions:
            result["reason"] = "head_excluded"
            result["route"] = "llm"
            result["guard_failures"].append(
                f"head_excluded:{dt} ({bundle.head_exclusions[dt]})")
            result["quality"]["exclusion_policy"] = "present"
            return result
        result["quality"]["exclusion_policy"] = (
            "present" if bundle.exclusion_policy else "absent")

        # #107 taxonomy-conformance projection: a canonical subclass the
        # head cannot express is projected (certificate_of_formation ->
        # other) or, when the head has no `other` (insurance), UNMAPPED ->
        # LLM.  Never force-fit into a sibling class.
        projected = project_subclass(bundle.maps, dt, sub)
        if sub is not None and projected is None:
            result["reason"] = "subclass_unmapped"
            result["route"] = SUBCLASS_UNMAPPED_ROUTE
            result["guard_failures"].append(
                f"subclass_unmapped:{dt}/{sub} (head has no `other`)")
            return result
        sub = projected

        # #103 catch-all guard: a prediction on a catch-all token (e.g.
        # `other`) is a "couldn't express it" signal — it fails the fast
        # path REGARDLESS of authentic support (corporate_record.other = 5
        # at the tier floor still routes LLM).  Checked before the support
        # gate: catch-all is the stronger claim.
        if sub in CATCH_ALL_LABELS:
            result["reason"] = "catchall_label"
            result["route"] = "llm"
            result["guard_failures"].append(f"catchall_label:{dt}/{sub}")
            return result

        # authentic support gate: support counts from the bundle sidecar
        counts = bundle.support_counts.get(dt, {})
        support = counts.get(sub, 0) if sub is not None else 0
        if dt in bundle.maps and sub is not None and support < min_authentic_support:
            result["reason"] = "insufficient_support"
            result["route"] = "llm"
            result["guard_failures"].append(
                f"authentic support {support} < {min_authentic_support} "
                f"for {dt}/{sub}")
            return result

        p_dt = merged["calibrated_confidence"]
        p_sc = merged["subclass_confidence"]
        agree = merged["agreement"]
        margin = merged["margin"]
        need_subclass = sub is not None
        fast_path = (
            dt != ABSTAIN_UNKNOWN_CLASS
            and p_dt >= doc_confidence
            and (not need_subclass or p_sc >= subclass_confidence)
            and agree >= window_agreement
            and margin >= margin_gate
            and agree >= agreement_gate
        )
        result["route"] = "fast_path" if fast_path else "llm"
        result["reason"] = "fast_path" if fast_path else "gate_fail"
        result["quality"].update({
            "sections_ok": True,  # short forms: empty-by-design (P5)
            "triage_vocab_ok": vocab_ok,
            "coverage": 1.0,  # no truncation — encode raises on overflow
        })
        if not fast_path:
            failures = []
            if p_dt < doc_confidence:
                failures.append(f"doc_confidence {p_dt} < {doc_confidence}")
            if need_subclass and p_sc < subclass_confidence:
                failures.append(f"subclass_confidence {p_sc} < {subclass_confidence}")
            if agree < max(window_agreement, agreement_gate):
                failures.append(f"agreement {agree} < {max(window_agreement, agreement_gate)}")
            if margin < margin_gate:
                failures.append(f"margin {margin} < {margin_gate}")
            result["guard_failures"].extend(failures)
        return result
    except Exception as exc:  # noqa: BLE001 — fail-open: never raise to caller
        return {
            **result,
            "status": "failure",
            "route": "llm",
            "reason": "bert_error",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_point": _tb.format_exception_only(type(exc), exc)[-1].strip(),
            },
            "guard_failures": result["guard_failures"] + ["bert_error"],
        }


# Typed result alias for readability (kept as plain dict for span serialization).
TriageResult = dict[str, Any]
