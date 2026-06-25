#!/usr/bin/env python3
"""Vendor omega_one's numpy-only TRUST subset into HazardPulse.

HazardPulse's measured failure is calibration/trust, not raw accuracy. The
omega_one project (sibling repo) already implements a measured, hardened trust
layer: split-conformal prediction, Mahalanobis OOD, selective prediction
(risk-coverage / abstention), an immune-mode guardian, and Ed25519-signed,
independently re-runnable decision receipts. Rather than reimplement a thin /
buggy copy here, we VENDOR the exact modules.

Why vendor instead of depend:
  - HazardPulse CI is its own repo (`pip install -e .` on GitHub Actions). A
    path/published dependency on omega_one would drag its 88 MB `.cl` toolchain
    and a cross-repo checkout into every scoring run.
  - The TRUST subset we need is pure-numpy + stdlib (+ lazy `cryptography` only
    for signature *verification*). It is a small, closed import graph (verified:
    trusted -> {conformal, ood_selector, batchsign}; regression -> trusted;
    guardian -> _guards; the rest are leaves). Vendoring gives deterministic,
    version-pinned CI with no heavy toolchain.

Provenance is recorded in VENDOR_MANIFEST.json (source repo + commit + per-file
sha256) so drift is detectable: `python scripts/vendor_omega_trust.py --verify`
fails if the vendored bytes no longer match the manifest.

Usage:
    python scripts/vendor_omega_trust.py            # (re)vendor from source
    python scripts/vendor_omega_trust.py --verify   # check vendored == manifest
    OMEGA_ONE_ROOT=/path/to/omega_one python scripts/vendor_omega_trust.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "src" / "hazardpulse" / "trust" / "_vendor_omega"
MANIFEST_PATH = DEST / "VENDOR_MANIFEST.json"

# The closed import graph of the numpy-only trust subset (Phase 1).
# Order is leaves-first for readability; copy order does not matter.
MODULES = (
    "_guards.py",
    "metrics.py",
    "selective.py",
    "conformal.py",
    "ood_selector.py",
    "batchsign.py",
    "trusted.py",
    "regression.py",
    "guardian.py",
)

# The public API the trust layer re-exports (kept in lockstep with omega_one's
# own __all__ for these modules). Used to generate _vendor_omega/__init__.py.
REEXPORTS = {
    "conformal": ["ConformalPredictor", "MondrianConformalPredictor", "coverage", "group_coverage"],
    "ood_selector": ["MahalanobisOOD", "OODSelector"],
    "selective": ["aurc", "ece", "risk_coverage", "selective_report",
                  "threshold_for_coverage", "threshold_for_risk"],
    "trusted": ["TrustedDecision", "fast_trusted_decision", "load_ed25519_pubkey",
                "verify_trusted_receipt", "verify_trusted_receipt_modes"],
    "regression": ["ConformalRegressor", "CQRRegressor", "AdaptiveConformalRegressor",
                   "TrustedRegression", "regression_coverage", "verify_regression_receipt"],
    "guardian": ["CoherenceGuardian", "GatewayMode"],
    "batchsign": ["sign_batch", "sign_batch_full", "verify_batch_signature",
                  "merkle_root", "merkle_proof", "verify_merkle_proof"],
}


def _source_root() -> Path:
    env = os.environ.get("OMEGA_ONE_ROOT")
    if env:
        root = Path(env)
        return root / "omega" if (root / "omega").is_dir() else root
    # Default: sibling layout  Projects/hazardpulse  and  Projects/Coherence/omega_one
    return REPO_ROOT.parent / "Coherence" / "omega_one" / "omega"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _render_init(source_commit: str | None) -> str:
    lines = [
        '"""VENDORED from omega_one (do NOT edit by hand).',
        "",
        f"Source: omega_one @ {source_commit or 'unknown'}",
        "Re-vendor with: python scripts/vendor_omega_trust.py",
        "Verify drift with: python scripts/vendor_omega_trust.py --verify",
        "",
        "The numpy-only trust subset: split-conformal prediction, Mahalanobis OOD,",
        "selective prediction, the immune-mode guardian, and Ed25519-signed",
        "re-runnable decision receipts. Pure numpy + stdlib (cryptography is imported",
        "lazily, only for signature verification).",
        '"""',
        "from __future__ import annotations",
        "",
    ]
    for mod, names in REEXPORTS.items():
        joined = ", ".join(names)
        lines.append(f"from .{mod} import {joined}")
    lines.append("")
    all_names = [n for names in REEXPORTS.values() for n in names]
    lines.append("__all__ = [")
    for n in all_names:
        lines.append(f'    "{n}",')
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def vendor() -> int:
    src_root = _source_root()
    if not src_root.is_dir():
        print(f"ERROR: omega_one source not found at {src_root}. "
              f"Set OMEGA_ONE_ROOT to the omega_one checkout.", file=sys.stderr)
        return 2
    missing = [m for m in MODULES if not (src_root / m).is_file()]
    if missing:
        print(f"ERROR: source modules missing in {src_root}: {missing}", file=sys.stderr)
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(src_root.parent)
    manifest_files: dict[str, str] = {}
    for m in MODULES:
        data = (src_root / m).read_bytes()
        (DEST / m).write_bytes(data)
        manifest_files[m] = _sha256(data)

    (DEST / "__init__.py").write_text(_render_init(commit), encoding="utf-8")

    manifest = {
        "vendored_from": "omega_one",
        "source_path": str(src_root),
        "source_commit": commit,
        "vendored_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modules": MODULES,
        "sha256": manifest_files,
        "note": "Re-vendor via scripts/vendor_omega_trust.py; verify with --verify.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(MODULES)} trust modules from omega_one @ {commit} -> {DEST}")
    for m in MODULES:
        print(f"  {m}  sha256={manifest_files[m][:12]}..")
    return 0


def verify() -> int:
    if not MANIFEST_PATH.is_file():
        print("ERROR: no VENDOR_MANIFEST.json; run without --verify to vendor first.",
              file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("sha256", {})
    drift = []
    for m in manifest.get("modules", MODULES):
        path = DEST / m
        if not path.is_file():
            drift.append(f"{m}: MISSING")
            continue
        got = _sha256(path.read_bytes())
        if got != expected.get(m):
            drift.append(f"{m}: sha256 mismatch (got {got[:12]}.., expected {str(expected.get(m))[:12]}..)")
    if drift:
        print("VENDOR DRIFT DETECTED:", file=sys.stderr)
        for d in drift:
            print(f"  - {d}", file=sys.stderr)
        return 1
    print(f"Vendored trust subset matches manifest "
          f"({len(manifest.get('modules', []))} modules, omega_one @ "
          f"{manifest.get('source_commit')}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check vendored files match VENDOR_MANIFEST.json (no copy).")
    args = parser.parse_args(argv)
    return verify() if args.verify else vendor()


if __name__ == "__main__":
    raise SystemExit(main())
