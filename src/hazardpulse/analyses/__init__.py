"""Cross-modality population-scale analyses.

Each module here implements one falsifiable hypothesis from the
LAIC (Lithosphere-Atmosphere-Ionosphere Coupling) and lightning/cyclone
literature. They follow Signalbook's analysis pattern: a dataclass
holding the result + a ``run_*`` function that takes a corpus and
returns the populated dataclass.

Available analyses (run independently or as a suite):

    run_earthquake_geomagnetic_precursor   # Sobolev / Kp pre-quake
    run_earthquake_solar_flare_precursor   # Freund / X-ray pre-quake
    run_earthquake_imf_bz_precursor        # IMF Bz southward pre-quake
    run_hurricane_lightning_correlation    # Fierro+ GLM-RI link
    run_tornado_lightning_leadup           # Steiger+ flash jump pre-tornado
    run_cme_hurricane_intensification      # Thakur+ CME-RI link
"""
from hazardpulse.analyses.earthquake_precursors import (
    EarthquakeGeomagneticResult,
    EarthquakeSolarFlareResult,
    EarthquakeImfBzResult,
    run_earthquake_geomagnetic_precursor,
    run_earthquake_solar_flare_precursor,
    run_earthquake_imf_bz_precursor,
)
from hazardpulse.analyses.lightning_correlations import (
    HurricaneLightningResult,
    TornadoLightningResult,
    run_hurricane_lightning_correlation,
    run_tornado_lightning_leadup,
)
from hazardpulse.analyses.cme_hurricane import (
    CmeHurricaneResult,
    run_cme_hurricane_intensification,
)

__all__ = [
    "EarthquakeGeomagneticResult",
    "EarthquakeSolarFlareResult",
    "EarthquakeImfBzResult",
    "HurricaneLightningResult",
    "TornadoLightningResult",
    "CmeHurricaneResult",
    "run_earthquake_geomagnetic_precursor",
    "run_earthquake_solar_flare_precursor",
    "run_earthquake_imf_bz_precursor",
    "run_hurricane_lightning_correlation",
    "run_tornado_lightning_leadup",
    "run_cme_hurricane_intensification",
]
