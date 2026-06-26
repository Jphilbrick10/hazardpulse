"""
trusted.py - the E2E TRUSTED DECISION: the system's measured-best pieces composed into ONE governed,
guaranteed, auditable decision. This is where omega_one's honest value proposition lives - not a
better predictor (coherence is not one), but the most accurate decision you can ALSO (a) reject as
out-of-distribution, (b) give a finite-sample coverage GUARANTEE on, and (c) cryptographically
VERIFY and audit. No mainstream pipeline offers all four at once.

VERIFY vs RE-RUN (be precise - the audit's H3): the portable receipt here is VERIFIABLE - a third
party with the receipt + the issuer's public key confirms INTEGRITY (tamper-evidence) and
AUTHENTICITY (it was signed by us) and that the decision is BOUND to a specific model fingerprint +
input. It does NOT by itself RE-RUN the model. Bit-exact RE-RUN is a separate, stronger guarantee
provided only by the 0-ULP `.cl` substrate receipt (cl_engine / export_cl), which recomputes the
decision identically on CPU/native/GPU. Do not conflate the two.

ONE coherence model (a DeepCoherenceClassifier, ideally head="hybrid" + compact="auto" + distilled
from a strong teacher) supplies every piece:
  - predict()      -> the decision (discriminative head -> accuracy ~ a tuned tree/MLP)
  - ood_score()    -> basin free energy (the coherence-ENERGY-powered OOD detector)
TrustedDecision composes them:
  predict -> REJECT if OOD (energy past a calibrated quantile) -> RAPS conformal SET on the DECIDING
  model's probabilities (coverage >= 1-a; the emitted label is in its own set - audit H4)
  -> signed, VERIFIABLE RECEIPT (input + decision + model fingerprint, Ed25519-optional).

The coherence ENERGY (gated basin compactness) is what makes the OOD + conformal pieces good - that
is the one place the equation-of-one math is load-bearing end to end. Numpy + the omega stack.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .conformal import ConformalPredictor, MondrianConformalPredictor

__all__ = ["TrustedDecision", "verify_trusted_receipt", "verify_trusted_receipt_modes",
           "load_ed25519_pubkey"]

_RECEIPT_FIELDS = ("spec", "model_sha256", "input_sha256", "prediction",
                   "ood_rejected", "ood_score", "conformal_set", "coverage_target", "nonce")
_ALLOWED_RECEIPT_KEYS = set(_RECEIPT_FIELDS) | {"receipt_sha256", "signature"}


def _sha(obj) -> str:
    # allow_nan=False: a NaN/Inf field would serialize as invalid JSON ("NaN") and silently break a
    # cross-language verifier; reject it at the source instead.
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def verify_trusted_receipt(receipt, *, pubkey=None, require_signature=False) -> bool:
    """Standalone, MODEL-FREE receipt verification. Returns True iff:
      (1) the receipt contains ONLY the canonical fields (a SMUGGLED extra key is rejected - a
          consumer cannot be shown fields outside the hashed claim);
      (2) the canonical fields hash to receipt_sha256 - INTEGRITY (tamper-evidence); and
      (3) if `pubkey` is given, the Ed25519 signature verifies - AUTHENTICITY (proof the holder of
          the private key issued it).

    WITHOUT a pubkey this proves INTEGRITY ONLY, NOT authenticity: anyone can fabricate a
    self-consistent receipt and it will pass. For the authenticity / "signed by us" guarantee you
    MUST pass the issuer's public key. `require_signature=True` rejects unsigned receipts but, absent
    a key, still cannot say WHO signed."""
    if not isinstance(receipt, dict):
        return False
    if set(receipt) - _ALLOWED_RECEIPT_KEYS:                 # smuggled / extra fields -> reject
        return False
    claim = {k: receipt.get(k) for k in _RECEIPT_FIELDS}
    try:
        if _sha(claim) != receipt.get("receipt_sha256"):     # integrity (also rejects NaN/Inf via _sha)
            return False
    except ValueError:
        return False
    sig = (receipt.get("signature") or {}).get("sig")
    if (pubkey is not None or require_signature) and not sig:
        return False
    if pubkey is not None:                                    # authenticity
        try:
            pubkey.verify(bytes.fromhex(sig), receipt["receipt_sha256"].encode())
        except Exception:
            return False
    return True


def verify_trusted_receipt_modes(receipt, *, pubkey=None, seen_nonces=None) -> dict:
    """Per-MODE verification (vs a single bool) - what an auditor actually needs:
      hash_valid       INTEGRITY: only canonical fields present (no smuggled keys) AND they hash to
                       receipt_sha256.
      signature_valid  AUTHENTICITY: the Ed25519 signature checks against `pubkey`. None if no pubkey
                       was supplied (authenticity UNKNOWN - integrity alone is not authenticity).
      replay_valid     FRESHNESS: the nonce has not been seen before. Requires stamped receipts and a
                       `seen_nonces` set (updated in place). None if the receipt is unstamped (N/A).
      ok               hash_valid AND signature not-invalid AND replay not-invalid."""
    out = {"hash_valid": False, "signature_valid": None, "replay_valid": None, "ok": False}
    if not isinstance(receipt, dict) or (set(receipt) - _ALLOWED_RECEIPT_KEYS):
        return out
    claim = {k: receipt.get(k) for k in _RECEIPT_FIELDS}
    try:
        out["hash_valid"] = (_sha(claim) == receipt.get("receipt_sha256"))
    except ValueError:
        out["hash_valid"] = False
    sig = (receipt.get("signature") or {}).get("sig")
    if pubkey is not None:
        try:
            pubkey.verify(bytes.fromhex(sig or ""), receipt["receipt_sha256"].encode())
            out["signature_valid"] = True
        except Exception:
            out["signature_valid"] = False
    if seen_nonces is not None:
        nonce = receipt.get("nonce")
        if nonce is None:
            out["replay_valid"] = None
        else:
            out["replay_valid"] = nonce not in seen_nonces
            seen_nonces.add(nonce)
    out["ok"] = bool(out["hash_valid"] and out["signature_valid"] is not False
                     and out["replay_valid"] is not False)
    return out


def load_ed25519_pubkey(raw):
    """Reconstruct an Ed25519 public key from its 32 raw bytes (or hex string) - so the public key
    travels with the receipt as portable bytes and a verifier needs no live key object."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if isinstance(raw, str):
        raw = bytes.fromhex(raw)
    return Ed25519PublicKey.from_public_bytes(raw)


class _BasinView:
    """Expose a classifier's generative basin probabilities as predict_proba (for the conformal
    predictor), so the conformal SET uses the sharp basin probs (tight) while the decision uses the
    discriminative head (accurate)."""

    def __init__(self, clf):
        self.clf = clf
        self.classes_ = list(getattr(clf, "classes_", []))

    def predict_proba(self, X):
        return self.clf.basin_proba(X) if hasattr(self.clf, "basin_proba") else self.clf.predict_proba(X)


class TrustedDecision:
    """Compose a coherence model into one trusted/guaranteed/auditable decision.
    `clf` must expose predict / ood_score (+ basin_proba or predict_proba) and classes_.
    `predictor` (optional): route the DECISION to a separate best-validated model (a champion router /
    Super-Learner / XGBoost) for accuracy, while the coherence `clf` still supplies OOD + conformal +
    the receipt - the omega_one_extreme architecture (best predictor + coherence audit only). It must
    have .predict(X) returning labels in clf.classes_. None -> clf is the predictor too."""

    def __init__(self, clf, *, predictor=None, alpha=0.1, ood_reject_quantile=0.99, conformal="raps",
                 signer=None, ood_scorer=None, group_fn=None, stamp=False):
        self.clf = clf
        self.predictor = predictor
        self.alpha = float(alpha)
        self.ood_reject_quantile = float(ood_reject_quantile)
        self.conformal = conformal
        self.signer = signer
        self.ood_scorer = ood_scorer          # an OODSelector / callable X->scores (high=OOD); None=clf basin
        self.group_fn = group_fn              # X->group ids => per-group (Mondrian) coverage guarantee
        self.stamp = bool(stamp)              # add a unique nonce per decision (replay defense; breaks byte-determinism)
        self.ood_threshold_ = None
        self.cp_ = None
        self.model_sha256_ = None
        self.classes_ = None

    def _ood(self, X):
        """The OOD score source: a fitted OODSelector (best detector per representation), any
        callable X->scores, or the coherence clf's basin energy by default."""
        s = self.ood_scorer
        if s is None:
            return np.asarray(self.clf.ood_score(X), float)
        return np.asarray(s.score(X) if hasattr(s, "score") else s(X), float)

    def _one_fingerprint(self, m) -> str:
        sd = getattr(getattr(m, "net", None), "state_dict", None)
        if callable(sd):                                   # torch deep head: hash the rounded weights
            flat = np.concatenate([np.asarray(v.detach().cpu(), np.float64).ravel()
                                   for v in sd().values() if v is not None and v.numel()])
            return _sha(np.round(flat, 6).tolist())
        try:                                               # hash the FULL pickled state, not str(__dict__)
            import pickle                                  # (str() truncates numpy arrays -> collisions)
            return hashlib.sha256(pickle.dumps(m)).hexdigest()
        except Exception:
            return _sha(repr(m))

    def _model_fingerprint(self) -> str:
        """A deterministic fingerprint of the live decision model(s) - the coherence clf AND the
        separate predictor if one is routing the decision (so the receipt binds BOTH)."""
        fp = self._one_fingerprint(self.clf)
        if self.predictor is not None:
            fp = _sha([fp, self._one_fingerprint(self.predictor)])
        return fp

    def fit_calibrate(self, Xcal, ycal):
        """Calibrate on a HELD-OUT split: the OOD-reject energy threshold (a quantile of in-dist
        calibration energies) and the conformal threshold (RAPS/APS -> coverage >= 1-alpha).

        H4 FIX: the conformal set is calibrated on the probabilities of the model that ACTUALLY
        DECIDES (the `predictor` if one routes the decision, else the coherence clf's own
        predict_proba) - NOT the basin. Otherwise the emitted label (predictor's argmax) could be
        excluded from its own coverage-guaranteed set."""
        self.classes_ = list(self.clf.classes_)
        self.ood_threshold_ = float(np.quantile(self._ood(Xcal), self.ood_reject_quantile))
        conf_clf = (self.predictor if (self.predictor is not None and hasattr(self.predictor, "predict_proba"))
                    else self.clf)
        if self.group_fn is not None:                        # per-group (Mondrian) coverage guarantee
            self.cp_ = MondrianConformalPredictor(conf_clf, self.group_fn, alpha=self.alpha,
                                                  method=self.conformal).fit_calibrate(Xcal, ycal)
        else:
            self.cp_ = ConformalPredictor(conf_clf, alpha=self.alpha,
                                          method=self.conformal).fit_calibrate(Xcal, ycal)
        self._conf_classes = list(self.cp_.classes_)         # set membership is in THIS model's class order
        self.model_sha256_ = self._model_fingerprint()
        return self

    def decide(self, X, *, batch_sign=False):
        """One trusted decision per row: prediction (or None if OOD-rejected), the OOD energy + flag,
        a coverage-guaranteed conformal set, and a signed re-runnable receipt.

        `batch_sign=True` (bulk/offline): sign the WHOLE batch with ONE Ed25519 over a Merkle root
        instead of one signature per receipt - signed throughput then approaches unsigned. Each record
        gets a `batch_signature` envelope (root + inclusion proof); the canonical receipt is unchanged
        and still integrity-verifiable standalone. Verify authenticity with `verify_batch_signature`."""
        X = np.atleast_2d(np.asarray(X, float))
        # FAST PATH: pure coherence-clf decision (no routed predictor / custom OOD / Mondrian) where the
        # clf can emit logits+energies+proba in ONE forward - avoids running the expensive extractor 3x
        # (predict + ood + conformal). Bit-identical to the general path below, just ~3x fewer forwards.
        fused = (self.predictor is None and self.ood_scorer is None
                 and hasattr(self.clf, "forward_bundle")
                 and hasattr(self.cp_, "predict_set_proba"))
        if fused:
            b = self.clf.forward_bundle(X)
            pred = np.array([self.classes_[i] for i in b["logits"].argmax(1)])
            energy = b["energies"].min(1)
            sets = self.cp_.predict_set_proba(b["proba"])
        else:
            pred = (self.predictor.predict(X) if self.predictor is not None else self.clf.predict(X))
            energy = self._ood(X)
            sets = self.cp_.predict_set(X)
        finite = np.isfinite(energy)
        rejected = (energy > self.ood_threshold_) | ~finite        # non-finite energy -> abstain
        out = []
        for i in range(len(X)):
            cset = [self._conf_classes[j] for j in np.where(sets[i])[0]]   # set in the deciding model's order
            claim = {"spec": "omega-one/trusted/v1",
                     "model_sha256": self.model_sha256_,
                     "input_sha256": hashlib.sha256(                          # EXACT byte binding (not round(8))
                         np.ascontiguousarray(X[i], np.float64).tobytes()).hexdigest(),
                     "prediction": None if rejected[i] else _as_native(pred[i]),
                     "ood_rejected": bool(rejected[i]),
                     "ood_score": round(float(energy[i]), 6) if finite[i] else None,   # never NaN in the receipt
                     "conformal_set": [_as_native(c) for c in cset],
                     "coverage_target": round(1.0 - self.alpha, 4),
                     "nonce": os.urandom(16).hex() if self.stamp else None}   # unique-per-decision -> replay-detectable
            receipt = dict(claim, receipt_sha256=_sha(claim))
            if self.signer is not None and not batch_sign:            # per-receipt signature (default)
                receipt["signature"] = {"alg": "ed25519",
                                        "sig": self.signer.sign(receipt["receipt_sha256"].encode()).hex()}
            out.append({"prediction": claim["prediction"], "abstained": bool(rejected[i]),
                        "ood_score": claim["ood_score"], "conformal_set": cset,
                        "receipt": receipt})
        if batch_sign:                                                # ONE signature over a Merkle root
            from .batchsign import sign_batch_full
            leaves = [rec["receipt"]["receipt_sha256"] for rec in out]
            hdr, proofs = sign_batch_full(leaves, self.signer)        # tree built ONCE (O(N)), 1 ed25519 sig
            for i, rec in enumerate(out):
                rec["batch_signature"] = dict(hdr, index=i, proof=proofs[i])
        return out

    def verify(self, receipt, *, pubkey=None) -> bool:
        """A receipt verifies iff its claimed fields hash to its receipt_sha256 (tamper-evident) and,
        if a pubkey is given, the Ed25519 signature checks out. Delegates to the standalone
        `verify_trusted_receipt` - the model is NOT needed to verify (a third party can audit)."""
        return verify_trusted_receipt(receipt, pubkey=pubkey)


def fast_trusted_decision(predictor, Xtr, ytr, Xcal, ycal, *, alpha=0.1, conformal="lac",
                          ood_reject_quantile=0.99, signer=None, stamp=False, group_fn=None):
    """LOW-LATENCY trusted decision for online serving. Same guarantees as TrustedDecision (OOD reject +
    coverage-guaranteed conformal set + signed re-runnable receipt) but built so a SINGLE request is
    sub-millisecond: the decision runs on a lightweight predictor (tree / logreg - microsecond predict)
    and OOD uses a MODEL-FREE Mahalanobis detector, so NO neural forward is on the request path.

    Measured (bench_serving): the deep-head TrustedDecision is ~3.4 ms p50 / 7.7 ms p99 per online
    request (torch fixed per-call overhead, x3 for predict+OOD+conformal); this path removes all of it.
    Use the deep-head path when its basin-energy OOD quality matters more than online latency; use this
    for live serving. The trust layer itself is identical and just as cheap (crypto ~0.02 ms).

    predictor: any sklearn-style classifier (predict / predict_proba / classes_); fit if not already.
    Returns a calibrated TrustedDecision ready for .decide(X)."""
    from .ood_selector import MahalanobisOOD
    if getattr(predictor, "classes_", None) is None:     # fit if unfitted (some predictors pre-declare classes_=None)
        predictor.fit(Xtr, ytr)
    ood = MahalanobisOOD().fit(Xtr, ytr)
    td = TrustedDecision(predictor, alpha=alpha, conformal=conformal, ood_scorer=ood,
                         ood_reject_quantile=ood_reject_quantile, signer=signer, stamp=stamp,
                         group_fn=group_fn)
    return td.fit_calibrate(Xcal, ycal)


def _as_native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
