#!/usr/bin/env python3
"""Independently verify a HazardPulse forecast receipt — with zero trust in us.

A HazardPulse forecast carries a signed receipt: a canonical hash of the exact
decision (model fingerprint + input hash + calibrated probability + interval +
abstention) and an Ed25519 signature over that hash. This script lets *anyone*
confirm a published forecast is authentic and untampered WITHOUT running or
trusting any HazardPulse code: it is fully self-contained (Python stdlib + the
`cryptography` library only) and re-implements the published receipt spec.

What it checks per receipt:
  1. INTEGRITY  — the receipt's core fields re-hash to its `receipt_sha256`
                  (so any field edit is detected). Always checked.
  2. AUTHENTICITY — the Ed25519 signature verifies against the published public
                  key (so only the holder of the private key could have issued
                  it). Checked when a public key is supplied.

Usage:
    python scripts/verify_forecast.py --artifact dist/data/replay/eq_fcst_*.json \
        --pubkey-file dist/data/evidence/public-key.json
    python scripts/verify_forecast.py --receipt receipt.json --pubkey <hex>

Exit code 0 iff every receipt passes (integrity, plus authenticity when a key is
given); 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# The published v1 receipt spec: the exact fields hashed into receipt_sha256,
# in a fixed set (the canonical encoder sorts keys, so order here is irrelevant).
RECEIPT_CORE_FIELDS = (
    "spec", "model_version", "model_sha256", "input_sha256", "issued_at",
    "raw_probability", "probability", "confidence_lo", "confidence_hi",
    "uncertainty_class", "ood_score", "ood_flag", "abstained", "abstain_reason",
    "gateway_mode", "coverage_target",
)


def canonical_sha256(claim: dict) -> str:
    """SHA-256 of the canonical JSON encoding (sorted keys, compact, no NaN)."""
    encoded = json.dumps(claim, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_receipt(receipt: dict, pubkey_hex: str | None, *,
                   key_is_explicit: bool = True) -> tuple[bool, str]:
    """Return (ok, detail). ok requires integrity, plus a signature if a key was ASKED FOR.

    `key_is_explicit` distinguishes a key the caller SUPPLIED (`--pubkey` / `--pubkey-file`) from
    one this script DISCOVERED on disk. That difference is the whole point:

      - Explicit key: the caller is asserting "this artifact should be signed by this issuer."
        An unsigned receipt then fails, and must -- otherwise `--pubkey` could be silently ignored
        and an unsigned artifact would pass an authenticity check it never underwent.
      - Discovered key: a convenience so the common case needs no flag. It is NOT an assertion
        about the artifact, so an unsigned receipt falls back to integrity-only and says so.

    Conflating the two made the public verifier print `[FAIL] ... no signature present` and exit 1
    for every unsigned artifact, purely because a published key happened to sit in `dist/`. On a
    zero-trust verifier a stranger runs to check our claims, FAIL reads as TAMPERED. The artifact
    was fine; the verifier was answering a question nobody asked.
    """
    if not isinstance(receipt, dict):
        return False, "not a receipt object"
    claim = {k: receipt.get(k) for k in RECEIPT_CORE_FIELDS}
    try:
        recomputed = canonical_sha256(claim)
    except (ValueError, TypeError) as exc:
        return False, f"uncanonicalizable (NaN/Inf?): {exc}"
    if recomputed != receipt.get("receipt_sha256"):
        return False, "INTEGRITY FAIL — fields do not match receipt_sha256 (tampered)"
    sig = (receipt.get("signature") or {}).get("sig")
    if pubkey_hex and not sig and not key_is_explicit:
        return True, ("integrity OK (unsigned receipt; the published key was auto-discovered, "
                      "not requested — pass --pubkey to REQUIRE a signature)")
    if pubkey_hex:
        if not sig:
            return False, "no signature present but a public key was supplied"
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
            pub.verify(bytes.fromhex(sig), receipt["receipt_sha256"].encode())
        except Exception as exc:
            return False, f"AUTHENTICITY FAIL — signature does not verify: {exc}"
        return True, "integrity OK + signature verified"
    return True, "integrity OK (authenticity not checked — no public key supplied)"


def _receipts_from(obj: dict) -> list[dict]:
    """Pull all forecast receipts out of a replay artifact (cells or storms) or a bare receipt."""
    if isinstance(obj, dict) and "receipt_sha256" in obj and "spec" in obj:
        return [obj]
    out: list[dict] = []
    for key in ("active_cells", "storms"):
        for item in obj.get(key, []) or []:
            r = item.get("receipt") if isinstance(item, dict) else None
            if isinstance(r, dict) and r.get("receipt_sha256"):
                out.append(r)
    return out


DEFAULT_PUBKEY_FILE = Path(__file__).resolve().parents[1] / "dist" / "data" / "evidence" / "public-key.json"


def _load_pubkey_hex(args) -> tuple[str | None, bool]:
    """Returns (key_hex, explicit). `explicit` means the CALLER asked for this key.

    See `verify_receipt`: a requested key makes a signature mandatory, a discovered one does not.
    """
    if args.pubkey:
        return args.pubkey.strip(), True
    if args.pubkey_file:
        rec = json.loads(Path(args.pubkey_file).read_text(encoding="utf-8"))
        return rec.get("public_key_hex"), True
    if DEFAULT_PUBKEY_FILE.exists():
        rec = json.loads(DEFAULT_PUBKEY_FILE.read_text(encoding="utf-8"))
        print(f"using published signing key from {DEFAULT_PUBKEY_FILE} "
              f"(auto-discovered; unsigned receipts are still checked for integrity)")
        return rec.get("public_key_hex"), False
    return None, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", type=Path, help="A replay artifact JSON (cells or storms).")
    src.add_argument("--receipt", type=Path, help="A single receipt JSON.")
    parser.add_argument("--pubkey", default=None, help="Ed25519 public key (hex).")
    parser.add_argument("--pubkey-file", default=None,
                        help="JSON with public_key_hex (e.g. dist/data/evidence/public-key.json).")
    args = parser.parse_args(argv)

    pubkey_hex, key_is_explicit = _load_pubkey_hex(args)
    payload = json.loads((args.artifact or args.receipt).read_text(encoding="utf-8"))
    receipts = _receipts_from(payload)
    if not receipts:
        print("No forecast receipts found (forecast not trust-wrapped yet).")
        return 1

    n_ok = 0
    n_signed = 0
    for i, receipt in enumerate(receipts):
        ok, detail = verify_receipt(receipt, pubkey_hex, key_is_explicit=key_is_explicit)
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] receipt {i + 1}/{len(receipts)} "
              f"({receipt.get('spec')}): {detail}")
        n_ok += int(ok)
        n_signed += int(bool((receipt.get("signature") or {}).get("sig")))

    # Report what was ACTUALLY checked. The old line said "(with authenticity)" whenever a key was
    # present, including for receipts that carry no signature at all -- claiming a stronger check
    # than was performed, on the one tool whose job is to not do that.
    if pubkey_hex and n_signed == len(receipts):
        scope = "with authenticity"
    elif pubkey_hex and n_signed:
        scope = f"integrity; authenticity on {n_signed}/{len(receipts)} signed"
    else:
        scope = "integrity only"
    print(f"\n{n_ok}/{len(receipts)} receipts verified ({scope}).")
    return 0 if n_ok == len(receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
