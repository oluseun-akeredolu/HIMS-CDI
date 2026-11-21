# Supplement S1 | Four-gate Phase I reproducibility

## The four files to open first

| Easy name | Purpose | Provenance |
|---|---|---|
| `phase1.py` | Authoritative v0.2.23 generator for the reported results | Byte-identical to the longer-named source in `audit/original_S1.zip` |
| `results/` | Frozen output directory underlying Sections 4–5 and Table 11 | Every member byte-identical to the archived `rerun_corrected/` folder |
| `verify.py` | Read-only checker of the 12 manuscript numeric checks | Same numeric checks; reads `results/` and prints findings rather than rewriting archival files |
| `g3.py` and `g3_results/` | Separate subsequent G3 boundary candidate and stored outputs | Byte-identical to the original candidate source and output records |

Run `python verify.py` (with dependencies in `requirements.txt`). To generate fresh outputs without overwriting anything, run `python phase1.py --out fresh_main` and optionally `python g3.py --out fresh_g3`.

The historical v0.2.9 source is at `source/v029.zip`; the archived corrected candidate source ZIP is at `source/v023.zip`. The original, unaltered supplement is retained as `audit/original_S1.zip` for provenance checks; `FileMap.csv` supplies SHA-256 mappings for readable names. The original timestamp-bearing `manifest.json` files are intentionally preserved without rewriting their historical generated-by fields.

**Inference boundary:** S1 establishes controlled synthetic rule conformance, not validated healthcare effectiveness or independent clinical assessment. The separate G3 candidate was not used to generate the reported 240-case results.
