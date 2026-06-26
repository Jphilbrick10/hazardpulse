"""Tiny dependency-free SVG renderers for the public trust scoreboard.

A reliability diagram is the single most honest picture of a probabilistic
forecaster: it plots predicted probability (x) against observed frequency (y).
A perfectly calibrated forecaster sits on the diagonal; HazardPulse's job is to
show its real curve, not hide it. This renders one as a self-contained inline
SVG (no JS, no chart library) so it drops straight into the static
``/verification`` page.
"""

from __future__ import annotations

import html

from .calibration import ReliabilityCurve

__all__ = ["reliability_diagram_svg"]


def _f(v: float) -> str:
    return f"{v:.1f}"


def reliability_diagram_svg(curve: ReliabilityCurve, *, size: int = 240, pad: int = 30,
                            title: str | None = None, accent: str = "#1976d2") -> str:
    """Render a reliability curve as an inline, accessible SVG string."""
    plot = size - 2 * pad
    x0 = y0 = pad

    def px(v: float) -> float:
        return x0 + max(0.0, min(1.0, v)) * plot

    def py(v: float) -> float:  # SVG y grows downward; 0 freq at the bottom
        return y0 + (1.0 - max(0.0, min(1.0, v))) * plot

    label = html.escape(title or "Reliability diagram")
    parts = [
        f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{label}" class="reliability-diagram">',
        f'<rect x="{x0}" y="{y0}" width="{plot}" height="{plot}" fill="none" '
        f'stroke="#ccc" stroke-width="1"/>',
        # perfect-calibration diagonal
        f'<line x1="{_f(px(0))}" y1="{_f(py(0))}" x2="{_f(px(1))}" y2="{_f(py(1))}" '
        f'stroke="#999" stroke-dasharray="4 3" stroke-width="1"/>',
    ]

    pts = [
        (b.mean_predicted, b.observed_freq, b.count)
        for b in curve.bins
        if b.count and b.mean_predicted == b.mean_predicted and b.observed_freq == b.observed_freq
    ]
    if pts:
        poly = " ".join(f"{_f(px(mp))},{_f(py(of))}" for mp, of, _ in pts)
        parts.append(
            f'<polyline points="{poly}" fill="none" stroke="{accent}" stroke-width="1.5"/>'
        )
        max_c = max(c for _, _, c in pts) or 1
        for mp, of, c in pts:
            r = 2.0 + 4.0 * (c / max_c) ** 0.5
            parts.append(
                f'<circle cx="{_f(px(mp))}" cy="{_f(py(of))}" r="{_f(r)}" '
                f'fill="{accent}" opacity="0.85"><title>predicted {mp:.2f}, '
                f'observed {of:.2f} (n={c})</title></circle>'
            )

    cx = x0 + plot / 2.0
    cy = y0 + plot / 2.0
    parts.append(
        f'<text x="{cx:.0f}" y="{size - 8}" text-anchor="middle" font-size="10" '
        f'fill="#666">Predicted probability</text>'
    )
    parts.append(
        f'<text x="12" y="{cy:.0f}" text-anchor="middle" font-size="10" fill="#666" '
        f'transform="rotate(-90 12 {cy:.0f})">Observed frequency</text>'
    )
    if title:
        parts.append(
            f'<text x="{x0}" y="{y0 - 12}" font-size="11" fill="#333">{html.escape(title)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
