\# HIMS-CDI - Phase I (CIC IoMT 2024 Benchmark)



Phase I empirical validation artifact. Frozen release: phase1-v1.0.



\## Results

ECE 0.0007 \[95% CI 0.0005-0.0010] | AUROC 0.9957 \[0.9952-0.9962]

F1 0.9953 \[0.9952-0.9955] | Brier 0.0062 \[0.0060-0.0063]



\## Reproduce

1\. Download the CIC IoMT 2024 dataset (72 CSV files) from the

&#x20;  Canadian Institute for Cybersecurity (UNB) and place them in this folder.

2\. pip install -r requirements.txt

3\. python final\_paper\_run.py



Sample: 1,812,537 rows (30,000/file). Peak memory: 4025.59 MB.

