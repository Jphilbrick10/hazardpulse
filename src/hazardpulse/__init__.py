"""HazardPulse: Unified natural hazard prediction from coherence field theory.

Three prediction systems derived from the Helmholtz coherence PDE:
    D * nabla^2(tau_c) - Gamma * tau_c + S = 0

- Earthquake: AUC 0.733 [0.704, 0.750] on 980 mainshocks (beats rate-only by +0.137)
- Hurricane RI: AUC 0.967 [0.955, 0.977] on NA best-track (beats persistence by +0.027)
- Tornado formation: AUC 0.644 [0.623, 0.658] day-ahead (no same-day data)
"""

__version__ = "0.1.0"
