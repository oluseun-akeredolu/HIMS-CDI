"""
real_source_fusion.py

Adapts the fusion architecture to what real CIC IoMT 2024 actually is:
ONE source (a single network-flow capture pipeline), not five.

Honest scope decision (per your instruction to proceed with option (a)):
  - The base classifier stands in for "per-source risk scoring."
  - The particle filter still does real work: temporal smoothing of the
    classifier's per-flow risk into a latent risk-state estimate. State
    dimension is 1 here (not 3) -- a vector state only makes sense with
    genuinely independent observation channels, and there's only one here.
    Using state_dim=3 fed by the same single scalar would repeat the exact
    "fake independence" problem already identified and fixed in the
    synthetic-data version; state_dim=1 is the honest choice for one source.
  - CorrelationAwareWeighting is retained but is INERT: with one source,
    there is nothing to correlate against, so it always returns weight=1.0.
    It's kept in the code path (not deleted) so the mechanism is visible
    and testable, but it does no real work on this dataset -- said plainly
    rather than silently disabled.
"""

from __future__ import annotations
import numpy as np


class ScalarParticleFilter:
    """Bootstrap particle filter, scalar latent risk state. N=1000,
    sigma=0.1, systematic resampling -- same numerical parameters as the
    spec, applied to a single-source scalar stream (see module docstring
    for why state_dim=1 here rather than the vector version)."""

    def __init__(self, n_particles: int = 1000, process_noise: float = 0.1, seed: int = 7):
        self.n = n_particles
        self.process_noise = process_noise
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.uniform(0, 1, self.n)
        self.weights = np.ones(self.n) / self.n

    def predict(self):
        noise = self.rng.normal(0, self.process_noise, self.n)
        self.particles = np.clip(self.particles + noise, 0, 1)

    def update(self, observation_risk: float, obs_noise: float = 0.15):
        likelihood = np.exp(-0.5 * ((observation_risk - self.particles) / obs_noise) ** 2)
        self.weights *= likelihood
        self.weights += 1e-300
        self.weights /= self.weights.sum()
        eff_n = 1.0 / np.sum(self.weights ** 2)
        if eff_n < self.n / 2:
            self._systematic_resample()

    def _systematic_resample(self):
        positions = (self.rng.uniform() + np.arange(self.n)) / self.n
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n) / self.n

    def estimate(self) -> float:
        return float(np.sum(self.particles * self.weights))

    def step(self, observation_risk: float) -> float:
        self.predict()
        self.update(observation_risk)
        return self.estimate()


def smooth_risk_sequence(raw_risk_scores: np.ndarray, pf: ScalarParticleFilter) -> np.ndarray:
    """Runs a whole sequence of per-flow raw risk scores through the
    particle filter in order, returning the smoothed sequence. State
    carries over across calls (pass the same `pf` across train/tune/test
    stages, exactly as the frozen-state design requires)."""
    out = np.empty(len(raw_risk_scores))
    for i, r in enumerate(raw_risk_scores):
        out[i] = pf.step(float(r))
    return out
