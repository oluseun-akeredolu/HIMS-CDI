#!/usr/bin/env python3
"""
Reference Implementation – Synthetic Data Generator for HIMS‑CDI

Generates 240 environments using the correlation structure from
the systematic review (`correlation_matrix.csv`).  The exact dataset
used in the evaluation was created with additional empirical marginal
transformations and is archived at Zenodo (DOI: 10.5281/zenodo.16897633).

Dependencies: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_ENVIRONMENTS = 240                    # total corpus size
N_INDICATORS   = 20

indicator_names = [
    "cti_modality_ind1", "cti_modality_ind2", "cti_modality_ind3", "cti_modality_ind4",
    "semantic_enrichment_ind1", "semantic_enrichment_ind2", "semantic_enrichment_ind3", "semantic_enrichment_ind4",
    "fusion_strategy_ind1", "fusion_strategy_ind2", "fusion_strategy_ind3", "fusion_strategy_ind4",
    "regulatory_traceability_ind1", "regulatory_traceability_ind2", "regulatory_traceability_ind3", "regulatory_traceability_ind4",
    "explainability_ind1", "explainability_ind2", "explainability_ind3", "explainability_ind4"
]

# Load the correlation matrix (provided in the repository)
corr_matrix = np.loadtxt("correlation_matrix.csv", delimiter=",")
assert corr_matrix.shape == (N_INDICATORS, N_INDICATORS), "Correlation matrix shape mismatch"

# Generate latent multivariate normal samples
latent = multivariate_normal.rvs(mean=np.zeros(N_INDICATORS),
                                 cov=corr_matrix,
                                 size=N_ENVIRONMENTS)

# Transform to uniform marginals using the standard normal CDF
uniform = 0.5 * (1 + np.erf(latent / np.sqrt(2)))

# NOTE: The actual generator applies empirical marginal transformations,
# label generation (DEPLOY_SYNTH), stratified splitting, and Gaussian noise
# perturbation.  This simplified version uses uniform margins as a placeholder.
# The complete pipeline is archived at the Zenodo DOI.

df = pd.DataFrame(uniform, columns=indicator_names)

# Save the full 240-environment dataset
df.to_csv("synthetic_hims_cdi_dataset.csv", index=False)

print("240-environment synthetic dataset saved: synthetic_hims_cdi_dataset.csv")
print("For the exact dataset used in the evaluation, refer to the Zenodo repository.")
