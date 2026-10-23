# Appendix F Item 5 — AHP source records unavailable

**Status:** Author clarification / claim withdrawal. This is not an AHP data file.

## Audit finding
The author reports that the available six files named as expert AHP matrices are byte-identical. The available matrix is therefore not evidence of six independently elicited expert judgments. As described in the audit notes, the supplied matrix yields the weight vector (0.125, 0.125, 0.250, 0.125, 0.250, 0.125), rather than the manuscript's reported vector (0.15, 0.15, 0.20, 0.15, 0.20, 0.15). The original individual matrices and authentic group-aggregation records could not be located.

**Do not present the identical matrices as authentic expert input. Do not claim that the reported weights, CR = 0.02, or Kendall's W = 0.82 have been verified from the available records.** If the underlying records cannot be found, withdraw the empirical AHP attribution.

## Proposed manuscript wording
> The original expert-level pairwise-comparison matrices and the record of group aggregation could not be located in the available study archive. The matrix files currently available are identical and do not establish independent expert elicitation; the weights and consistency statistics previously attributed to AHP are therefore not reported as verified empirical results. For the Phase I baseline configuration, the CDI uses the coefficient vector (0.15, 0.15, 0.20, 0.15, 0.20, 0.15) as a design configuration, not as an expert-validated priority structure. These coefficients should not be interpreted as evidence of expert consensus or clinical validity. Their effect on CDI scores and gate decisions must be assessed in a separately reported sensitivity analysis.

**Use the quoted coefficients only if the authors confirm these are the values actually used by the baseline code and dataset.** If a sensitivity audit shows that the existing values were not used consistently, correct the method and rerun affected results rather than preserving them for convenience.

## Supplementary-material action
Replace the purported raw-matrix files only after the archive is reviewed and versioned. One acceptable public artifact is a clear `ITEM5_AHP_RECORDS_UNAVAILABLE.md` disclosure, plus a description of the baseline coefficient vector and code that uses it. Do not upload blank templates as evidence of the original elicitation.

## New study option
The blank `templates/FUTURE_ONLY_AHP_matrix_template.csv` may be copied for a future, genuinely conducted expert elicitation. Any new elicitation must be dated and described as new data collection, with its panel, protocol, aggregation method, consistency checks, and analysis documented. It cannot retroactively validate the old claim.
