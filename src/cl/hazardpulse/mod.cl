// HazardPulse — Natural Hazard Intelligence Platform
// Written entirely in Coherence Lang
//
// The Equation of One all the way down:
//   S_One → Helmholtz PDE → Coherence Field → Tornado/Earthquake/Hurricane prediction
//   All computed in the language derived from the same unified theory.
//
// Modules:
//   tornado.gbt       — Gradient Boosted Tree inference
//   tornado.coherence  — Helmholtz PDE solver + coherence diagnostics
//   tornado.scorer     — Full scoring pipeline (fetch → compute → output)
//   earthquake.*       — Earthquake prediction (TODO)
//   hurricane.*        — Hurricane RI prediction (TODO)

module hazardpulse;

pub mod tornado {
    pub mod gbt;
    pub mod coherence;
    pub mod scorer;
}
