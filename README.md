# HIMS–CDI

Reference implementation of the HIMS–CDI scoring and evaluation framework.

## Features

- CDI computation with bootstrapped CIs
- Governance mapping and sensitivity analysis
- Multinomial Logistic Regression for threat prioritisation
- Baselines and ablation toggles
- Latency benchmarking
- Packaged for pip install

## Install

```bash
git clone https://github.com/oluseun-akeredolu/hims-cdi.git
cd hims-cdi

python3 -m venv .venv
source .venv/bin/activate   # for Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

## Usage

```bash
hims-cdi --data data/synthetic_hims_cdi_dataset.csv --output results.json
```

## Citation

Please cite the associated paper if you use this package.

## License

MIT
