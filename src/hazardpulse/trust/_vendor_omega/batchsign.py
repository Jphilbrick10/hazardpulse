"""
batchsign.py - aggregate signing: sign N decisions with ONE Ed25519 signature over a Merkle root instead
of one signature per decision. For BULK / offline signed scoring this removes the per-decision signing
cost (~20 us of ~32 us), so signed throughput approaches unsigned. Each receipt carries a compact
inclusion PROOF to the signed root and stays independently verifiable offline.

DESIGN (safe by construction): the canonical per-receipt format + the audited `verify_trusted_receipt`
are UNTOUCHED. The batch signature is a SEPARATE envelope attached alongside the receipt, never inside
it - so the receipt's smuggling guard and integrity hash are unaffected. Authenticity of a batch-signed
receipt = receipt integrity AND a valid Merkle proof from its hash to the root AND the root's signature.

Leaves and internal nodes are domain-separated (0x00 / 0x01 prefixes) for second-preimage resistance.
"""
import hashlib

__all__ = ["merkle_root", "merkle_proof", "verify_merkle_proof", "sign_batch", "sign_batch_full",
           "verify_batch_signature"]


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _leaf(hex_hash: str) -> bytes:
    return _h(b"\x00" + bytes.fromhex(hex_hash))


def _node(a: bytes, b: bytes) -> bytes:
    return _h(b"\x01" + a + b)


def _levels(leaf_hexes):
    if not leaf_hexes:
        raise ValueError("merkle: no leaves")
    level = [_leaf(h) for h in leaf_hexes]
    levels = [level]
    while len(level) > 1:
        level = [_node(level[i], level[i + 1] if i + 1 < len(level) else level[i])
                 for i in range(0, len(level), 2)]
        levels.append(level)
    return levels


def merkle_root(leaf_hexes) -> str:
    return _levels(leaf_hexes)[-1][0].hex()


def _proof_from_levels(levels, index):
    proof = []
    idx = index
    for level in levels[:-1]:
        sib = idx ^ 1
        if sib >= len(level):                 # an odd final node is duplicated with itself
            sib = idx
        proof.append(["R" if idx % 2 == 0 else "L", level[sib].hex()])
        idx //= 2
    return proof


def merkle_proof(leaf_hexes, index):
    """The sibling path from leaf `index` to the root: a list of [side, sibling_hex]. O(N) - builds the
    tree; for ALL proofs use `sign_batch_full` which builds the tree ONCE (O(N), not O(N^2))."""
    return _proof_from_levels(_levels(leaf_hexes), index)


def sign_batch_full(leaf_hexes, signer):
    """Build the tree ONCE and return (header, proofs) - the header has the signed root, `proofs[i]` is
    the inclusion proof for leaf i. This is the efficient path for signing a whole batch; calling
    `merkle_proof` per leaf would rebuild the tree each time (O(N^2))."""
    levels = _levels(leaf_hexes)
    root = levels[-1][0].hex()
    sig = signer.sign(root.encode()).hex() if signer is not None else None
    hdr = {"alg": "ed25519-merkle", "root": root, "n": len(leaf_hexes), "sig": sig}
    return hdr, [_proof_from_levels(levels, i) for i in range(len(leaf_hexes))]


def verify_merkle_proof(leaf_hex, proof, root_hex) -> bool:
    cur = _leaf(leaf_hex)
    for side, sib_hex in proof:
        sib = bytes.fromhex(sib_hex)
        cur = _node(cur, sib) if side == "R" else _node(sib, cur)
    return cur.hex() == root_hex


def sign_batch(leaf_hexes, signer):
    """Shared batch header: the Merkle root over the leaf hashes + ONE signature over that root."""
    root = merkle_root(leaf_hexes)
    sig = signer.sign(root.encode()).hex() if signer is not None else None
    return {"alg": "ed25519-merkle", "root": root, "n": len(leaf_hexes), "sig": sig}


def verify_batch_signature(receipt_sha256, batch_sig, *, pubkey=None) -> bool:
    """True iff the receipt's hash is included under the signed root (Merkle proof) and - if a pubkey is
    given - the root's Ed25519 signature verifies. Without a pubkey this proves only inclusion-integrity,
    not authenticity (same integrity-vs-authenticity split as the per-receipt verifier)."""
    if not isinstance(batch_sig, dict) or batch_sig.get("alg") != "ed25519-merkle":
        return False
    if not verify_merkle_proof(receipt_sha256, batch_sig.get("proof", []), batch_sig.get("root", "")):
        return False
    if pubkey is not None:
        s = batch_sig.get("sig")
        if not s:
            return False
        try:
            pubkey.verify(bytes.fromhex(s), batch_sig["root"].encode())
        except Exception:
            return False
    return True
