# HIMS-CDI | Reproducibility protocol (short-name release)

**Paper:** COSE-D-26-03217, revised four-gate specification. **Source of truth:** frozen `S1.zip/results/`, **not** `g3_results/` and not historical public CDI/CCR Zenodo archives.

## What the files mean

| File | Purpose |
|---|---|
| `S1.zip` | Corrected Phase I generator `phase1.py`, frozen main outputs `results/`, numeric verifier `verify.py` and separate G3 candidate `g3.py` |
| `S2.zip` | Controlled CTI/HTTPS fixture and offline replay evidence |
| `S3.docx` | Critical evidence synthesis (Tables F1–F3) |
| `S3_Data.zip` | Report-level evidence and linkage registers |
| `FileMap.csv` | Short name ↔ original path ↔ SHA-256 mapping |
| `SourceAudit.md` | Authoritative provenance and what is not validated |

The original S1 is retained **byte-for-byte** at `S1.zip/audit/original_S1.zip`. All relocated input and output members of `S1.zip` have unchanged SHA-256 values. The `verify.py` copy reads `results/` and prints results without overwriting historical JSON; all calculations and expected values remain unchanged. The original checker remains in the nested historical S1 archive. The frozen `results/manifest.json` was *not* rewritten and preserves its original generator metadata.

## On Windows PowerShell: read-only preflight

Open the extracted outer release folder and run:

```powershell
py -3 .\ReleaseCheck.py
```

This checks file and ZIP integrity, every alias against its byte-identical original, archive record counts, and historical versus candidate G3 fixture results. It uses only Python's standard library and never publishes or changes source data.

## On Windows PowerShell: numeric validation

```powershell
$root = 'D:\HIMS_CDI_REVIEW'
New-Item -ItemType Directory -Force -Path $root | Out-Null
Expand-Archive -LiteralPath .\S1.zip -DestinationPath (Join-Path $root 'S1') -Force
Set-Location (Join-Path $root 'S1')
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
& .\.venv\Scripts\python.exe .\verify.py
```

Expected numeric checks: **12/12** manuscript-oriented tests. The environment requires Python 3.12, numpy==2.3.5, pandas==2.2.3, scipy==1.17.0 and scikit-learn==1.8.0.

## Optional generator recreation (separate output paths)

```powershell
& .\.venv\Scripts\python.exe .\phase1.py --out (Join-Path $root 'new_main')
& .\.venv\Scripts\python.exe .\g3.py --out (Join-Path $root 'new_g3')
```

**Do not overwrite** `results/` or `g3_results/`. The Phase I figures in the manuscript are from the frozen `results/` files; the G3 candidate is not the generating source for those figures. Generator reruns have time-dependent manifests and may produce negligible platform-dependent floating-point representation differences; compare scientific outputs and documented tolerances, not unconditional byte identity for manifests. The script source SHA-256 values remain unchanged despite the shorter filenames.

## Separate S2 checks

```powershell
Expand-Archive -LiteralPath (Join-Path $PSScriptRoot 'S2.zip') -DestinationPath (Join-Path $root 'S2') -Force
Set-Location (Join-Path $root 'S2')
& ..\S1\.venv\Scripts\python.exe .\replay.py
& ..\S1\.venv\Scripts\python.exe .\taxii_exchange.py
```

The controlled HTTPS test requires an available OpenSSL executable and does not contact live healthcare organisations. The archived results record 10 HTTPS transactions and five receiver-to-gate scenarios; these are controlled fixtures, not deployment validation.

## Verification scope and limitations

- Main frozen results: 240 cases, 15 ELIGIBLE, 177 INELIGIBLE, 48 UNRESOLVED; 1,920 evidence rows, 7,200 execution rows.
- G3 fixture: archived expression 5/17 and later candidate 17/17. The candidate remains a separate, non-authoritative patch.
- Internal frozen scientific files: 14 matching outputs across the two archived result directories; `manifest.json` records distinct provenance.
- This release does **not** establish external independent HIMS ratings, clinical effectiveness, independent systematic literature screening, or public four-gate Zenodo DOI.
