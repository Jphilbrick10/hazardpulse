"""Small HTTP helpers with on-disk caching for reproducible data pulls."""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

DEFAULT_USER_AGENT = "hazardpulse/0.1 (+https://github.com/Jphilbrick10/hazardpulse)"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "http"


def _cache_path(url: str, namespace: str) -> Path:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / digest


def fetch_bytes(
    url: str,
    *,
    timeout: int = 60,
    namespace: str = "default",
    use_cache: bool = True,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> bytes:
    """Fetch bytes from a URL with simple file caching."""

    cache_path = _cache_path(url, namespace)
    if use_cache and cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
        data = resp.read()

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data


def fetch_text(
    url: str,
    *,
    timeout: int = 60,
    namespace: str = "default",
    encoding: str = "utf-8",
    errors: str = "replace",
    use_cache: bool = True,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch text from a URL with simple file caching."""

    data = fetch_bytes(
        url,
        timeout=timeout,
        namespace=namespace,
        use_cache=use_cache,
        refresh=refresh,
        user_agent=user_agent,
    )
    return data.decode(encoding, errors=errors)
