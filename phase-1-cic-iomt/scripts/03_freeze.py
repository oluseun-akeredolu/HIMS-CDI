import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import hashlib
import joblib
from datetime import datetime, timezone
from csra.config import ARTIFACTS_DIR


def main() -> None:
    bundle_path = ARTIFACTS_DIR / "pre_freeze_bundle_real.joblib"
    results_path = ARTIFACTS_DIR / "test_results_real.json"
    for p in (bundle_path, results_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run 02_fusion_calibration_evaluate.py first.")

    bundle = joblib.load(bundle_path)
    with open(results_path) as f:
        test_results = json.load(f)

    frozen = {
        "classifier": bundle["classifier"],
        "particle_filter_state": bundle["particle_filter_state"],
        "bbq_model": bundle["bbq_model"],
        "feature_cols": bundle["feature_cols"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_results_at_freeze_time": test_results,
    }

    frozen_path = ARTIFACTS_DIR / "frozen_model_real.joblib"
    joblib.dump(frozen, frozen_path)

    with open(frozen_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "artefact_path": str(frozen_path),
        "sha256": digest,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "dataset": "real CIC IoMT 2024 (systematic subsample, see subsample.py)",
    }
    manifest_path = ARTIFACTS_DIR / "frozen_model_real.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Frozen artefact:  {frozen_path}")
    print(f"SHA-256:          {digest}")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
