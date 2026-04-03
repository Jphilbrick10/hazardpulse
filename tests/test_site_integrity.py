from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_critical_public_pages_no_longer_contain_placeholder_content() -> None:
    pages = [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/hurricane/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "Hurricane Alice" not in text
        assert "TEST DATA" not in text
        assert "P(M6+ in 90 days)" not in text
        assert "hazardpulse.io" not in text


def test_key_public_pages_use_hazardpulse_branding() -> None:
    pages = [
        "dist/index.html",
        "dist/404.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/tornado/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "https://coherenceenergylabs.com" not in text, rel_path
        assert ">Coherence Energy Labs<" not in text, rel_path
        assert "Built with Coherence Lang" not in text, rel_path


def test_live_pulse_confidence_ranges_are_missing_or_sane() -> None:
    pulse = json.loads(_read_text("dist/data/live-pulse.json"))
    for hazard in pulse["hazards"]:
        lo = hazard.get("conf_lo")
        hi = hazard.get("conf_hi")
        probability = hazard.get("probability")
        if lo is None or hi is None:
            continue
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= probability <= hi


def test_public_license_asset_exists() -> None:
    assert (ROOT / "dist" / "COMMERCIAL_LICENSE.md").exists()


def test_remaining_dist_metadata_uses_primary_domain() -> None:
    for rel_path in [
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/methods/index.html",
        "dist/registry/index.html",
        "dist/ops/status/index.html",
        "dist/verification/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/tornado/index.html",
        "dist/legal/disclaimer/index.html",
        "dist/feed.xml",
        "dist/sitemap.xml",
        "dist/robots.txt",
    ]:
        assert "hazardpulse.io" not in _read_text(rel_path), rel_path


def test_key_public_pages_have_structural_basics() -> None:
    pages = [
        "dist/index.html",
        "dist/404.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "<title>" in text, rel_path
        assert '<meta name="description"' in text, rel_path
        assert '<link rel="canonical"' in text, rel_path
        assert '<meta property="og:title"' in text, rel_path
        assert '<meta name="twitter:title"' in text, rel_path
        assert 'class="skip-link"' in text, rel_path
        assert 'role="banner"' in text, rel_path
        assert 'id="main"' in text, rel_path
        assert 'role="contentinfo"' in text, rel_path
        assert len(re.findall(r"<h1\b", text)) == 1, rel_path


def test_key_public_pages_have_no_encoding_garbage() -> None:
    bad_fragments = ["ï¿½", "�", "Â·", "â€”", "â†’"]
    for rel_path in [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/legal/disclaimer/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
        "dist/404.html",
        "dist/feed.xml",
        "dist/sitemap.xml",
    ]:
        text = _read_text(rel_path)
        for fragment in bad_fragments:
            assert fragment not in text, (rel_path, fragment)


def test_worker_is_in_deploy_path() -> None:
    worker_path = ROOT / "src" / "worker.js"
    assert worker_path.exists(), "src/worker.js must exist for production deploys"
    wrangler_toml = _read_text("wrangler.toml")
    assert 'main = "./src/worker.js"' in wrangler_toml


def test_worker_api_smoke() -> None:
    result = subprocess.run(
        ["node", str(ROOT / "tests" / "worker_api_check.mjs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
