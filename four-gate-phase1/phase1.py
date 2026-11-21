#!/usr/bin/env python3
"""Generate a NEW, explicitly specified HIMS-CDI Phase I candidate simulation.

This does not reproduce the locked 19 October run and does not use or repair the
26 October output bundle. The synthetic dependence matrix and operational rules
below are a newly authored, transparent candidate specification.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
from scipy.stats import beta
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

CONFIG = {
  "run_id": "HIMS-CDI-PhaseI-candidate-UNKNOWN-fix-v0.2.23",
  "status": "newly specified synthetic Phase I run; not a reproduction of the historical run",
  "n_environments": 240,
  "n_indicators": 20,
  "n_controls": 6,
  "repetitions_per_environment": 30,
  "seeds": {"generation": 42, "dwell_time": 59, "scope": 65,
             "cv_fold_seed": 46, "measurement_noise": 42, "evidence_injection": 99},
  "measurement_noise_lognormal_sigma": 0.15,
  "base_evidence_degradation": {"missing": 0.05, "stale": 0.05, "unreviewed": 0.05},
  "runtime_thresholds": {"latency_p95_ms_strictly_less_than": 200.0,
                          "peak_memory_mb_strictly_less_than": 100.0},
  "gaussian_copula": {"construction": "factor model with one global factor plus six domain factors plus independent indicator noise",
                       "global_factor_variance": 0.05,
                       "domain_factor_variance": 0.20,
                       "idiosyncratic_variance": 0.75,
                       "implied_pairwise_correlations": {"same_domain": 0.25, "different_domains": 0.05},
                       "psd_by_construction": True},
  "copula_domain_assignments": {
    "D1": ["I01", "I02", "I03", "I04"], "D2": ["I05", "I06", "I07", "I08"],
    "D3": ["I09", "I10", "I11"], "D4": ["I12", "I13", "I14"],
    "D5": ["I15", "I16", "I17"], "D6": ["I18", "I19", "I20"]
  },
  "model_domain_rating_assignments": {
    "D1": ["I01", "I02", "I03", "I04"], "D2": ["I05", "I06", "I07", "I08"],
    "D3": ["I09", "I10", "I11"], "D4": ["I12", "I13", "I14"],
    "D5": ["I15", "I16", "I17"], "D6": ["I18", "I19", "I20"]
  },
  "synthetic_rule_assumptions": {
    "G1_scope_scenario": "Scope status is simulated from a fixed scenario distribution: PASS 90%, FAIL 5%, UNKNOWN 5%; this is an analyst-selected stress-test mix, not an empirical rate",
    "G2_control_status": "Each of six measured control proxies I01-I06 is PASS when >= 0.05 and FAIL when < 0.05; evidence degradation may turn PASS into UNKNOWN",
    "G3_runtime_status": "PASS only when measured p95 latency < 200 ms and maximum of 30 memory readings < 100 MB; any measured breach is FAIL",
    "G4_other_requirements": "REQ-A uses I17 >= 0.15; REQ-B uses I18 >= 0.15; measured value below threshold is FAIL; evidence degradation may turn PASS into UNKNOWN",
    "unknown_status": "UNKNOWN represents missing, stale, unreviewed, or unverifiable evidence; it is not used for a known numeric threshold failure",
    "conjunctive_gate_precedence": "FAIL if any mandatory subcondition is FAIL; otherwise UNKNOWN if any subcondition is UNKNOWN; otherwise PASS",
    "latency_baseline_ms": "clip(70 + 150*I19 + 20*I09 + 0.08*(dwell_days-62), 40, 280)",
    "memory_baseline_mb": "clip(30 + 45*I20 + 10*(1-I05) + 0.04*(dwell_days-62), 20, 95)",
    "measurement_noise": "30 per-environment latency and memory readings with mean-corrected lognormal multiplicative noise sigma=0.15",
    "evidence_injection": "For each PASS control or requirement item, missing/stale/unreviewed is injected with probability 0.05 each; FAIL states are unchanged",
    "eligibility": "G1 PASS AND G2 PASS AND G3 PASS AND G4 PASS; FAIL or UNKNOWN on any gate is not eligible"
  },
  "model_protocol": {
    "evaluation": "10 repeats of stratified 5-fold cross-validation; same folds for all comparators",
    "full_model_inputs": "one-hot status of G1 scope, six G2 controls, and two G4 requirements, plus latency p95 and peak memory; G3 is inferred from the raw resource values",
    "restricted_model_inputs": "D1-D6 mean indicator ratings only",
    "estimator": "StandardScaler + LogisticRegression(C=1.0, solver=lbfgs, max_iter=2000, class_weight=None)",
    "operating_threshold": "fixed at 0.50 before evaluation; no validation- or held-out-informed threshold selection",
    "equal_weight": "mean(D1-D6), divided by 5, fixed threshold 0.50",
    "oracle": "restate the observed rule-derived label; a tautological specification-identity check, not predictive evidence",
    "uncertainty_reporting": "mean, sample SD, minimum and maximum across ten CV repeats; split variability, not confidence intervals or independent external validation"
  }
}

COPULA_GROUPS = [list(range(0,4)), list(range(4,8)), list(range(8,11)), list(range(11,14)), list(range(14,17)), list(range(17,20))]
DOMAINS = COPULA_GROUPS
INDICATORS = [f"I{i:02d}" for i in range(1,21)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8', newline='\n')


def csv_write(path: Path, rows, fieldnames=None):
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False, lineterminator='\n', float_format='%.8f')
    else:
        rows = list(rows)
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            w.writeheader(); w.writerows(rows)


def make_correlation_matrix() -> np.ndarray:
    corr = np.full((20,20), CONFIG['gaussian_copula']['global_factor_variance'], dtype=float)
    np.fill_diagonal(corr, 1.0)
    for group in COPULA_GROUPS:
        for i in group:
            for j in group:
                if i != j:
                    corr[i,j] = (CONFIG['gaussian_copula']['global_factor_variance'] +
                                 CONFIG['gaussian_copula']['domain_factor_variance'])
    return corr


def generate_indicators(rng: np.random.Generator, n: int) -> np.ndarray:
    global_z = rng.normal(size=(n,1))
    domain_z = rng.normal(size=(n,6))
    epsilon = rng.normal(size=(n,20))
    domain_for_indicator = [next(d for d, group in enumerate(COPULA_GROUPS) if j in group) for j in range(20)]
    z = math.sqrt(0.05)*global_z + math.sqrt(0.20)*np.column_stack([domain_z[:,d] for d in domain_for_indicator]) + math.sqrt(0.75)*epsilon
    return np.clip(ndtr(z), 1e-6, 1-1e-6)


def safe_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel().tolist()
    sens = tp/(tp+fn) if tp+fn else float('nan')
    spec = tn/(tn+fp) if tn+fp else float('nan')
    precision = tp/(tp+fp) if tp+fp else 0.0
    f1p = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    f1n = 2*tn/(2*tn+fn+fp) if 2*tn+fn+fp else 0.0
    ba = (sens+spec)/2 if np.isfinite(sens) and np.isfinite(spec) else float('nan')
    return {"n": int(len(y_true)), "positive_n": int(np.sum(y_true)), "negative_n": int(len(y_true)-np.sum(y_true)),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "sensitivity": sens, "specificity": spec, "precision_positive": precision,
            "balanced_accuracy": ba, "f1_positive": f1p, "f1_negative": f1n,
            "macro_f1": (f1p+f1n)/2, "accuracy": (tp+tn)/len(y_true) if len(y_true) else float('nan')}


def conjunctive_status(states):
    """Three-valued conjunction: FAIL dominates; otherwise UNKNOWN; otherwise PASS."""
    states = list(states)
    if any(v == 'FAIL' for v in states):
        return 'FAIL'
    if any(v == 'UNKNOWN' for v in states):
        return 'UNKNOWN'
    if states and all(v == 'PASS' for v in states):
        return 'PASS'
    return 'UNKNOWN'


def eligible(g1, g2, g3, g4):
    return g1 == 'PASS' and g2 == 'PASS' and g3 == 'PASS' and g4 == 'PASS'


def make_scope_record(u):
    """Synthetic scope case: PASS 90%, known contradiction FAIL 5%, incomplete UNKNOWN 5%."""
    if u < 0.05:
        status='FAIL'; boundary=True; jurisdiction=True; registry=True; rationale=True; conflict=True
    elif u < 0.10:
        status='UNKNOWN'; boundary=True; jurisdiction=False; registry=True; rationale=False; conflict=False
    else:
        status='PASS'; boundary=True; jurisdiction=True; registry=True; rationale=True; conflict=False
    return {'status':status,'system_boundary_defined':boundary,'jurisdiction_resolved':jurisdiction,
            'control_registry_version_present':registry,'applicability_rationale_present':rationale,
            'known_scope_conflict':conflict}


def run(output_dir: Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"STOP: output directory is not empty: {output_dir}; nothing overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(output_dir/'config.json', CONFIG)

    # Generate Gaussian-copula indicators from the newly declared factor model.
    rng_gen=np.random.default_rng(CONFIG['seeds']['generation'])
    values=generate_indicators(rng_gen, CONFIG['n_environments'])
    corr=make_correlation_matrix()
    eig=np.linalg.eigvalsh(corr)
    corr_df=pd.DataFrame(corr, index=INDICATORS, columns=INDICATORS)
    corr_df.insert(0, 'indicator', INDICATORS)
    corr_df.to_csv(output_dir/'correlation_matrix.csv', index=False, lineterminator='\n', float_format='%.8f')

    # Dwell-time parameterised to arithmetic mean 62 and SD 28 days, as in the manuscript description.
    rng_dwell=np.random.default_rng(CONFIG['seeds']['dwell_time'])
    mean, sd = 62.0, 28.0
    sigma2=math.log(1+(sd/mean)**2)
    mu=math.log(mean)-sigma2/2
    dwell=rng_dwell.lognormal(mean=mu, sigma=math.sqrt(sigma2), size=CONFIG['n_environments'])
    env_ids=[f'ENV{i:03d}' for i in range(1, CONFIG['n_environments']+1)]

    # Derived ratings and simulated operational channels. I17 and I18 are declared gate-quality signals;
    # I19 and I20 are workload/resource drivers. These mappings are new assumptions, not recovered facts.
    D=np.column_stack([values[:, g].mean(axis=1) for g in DOMAINS])
    latency_base=np.clip(70 + 150*values[:,18] + 20*values[:,8] + 0.08*(dwell-62), 40, 280)
    memory_base=np.clip(30 + 45*values[:,19] + 10*(1-values[:,4]) + 0.04*(dwell-62), 20, 95)
    # Model scope explicitly as a synthetic record instead of using a numeric indicator proxy.
    # The 90/5/5 scope-status mix is an analyst-selected test scenario, not an estimate from real data.
    rng_scope=np.random.default_rng(CONFIG['seeds']['scope'])
    scope_records=[make_scope_record(float(u)) for u in rng_scope.random(len(env_ids))]
    clean_g1=np.array([r['status'] for r in scope_records])

    # G2: six mandatory control proxies. A measured numeric failure is FAIL.
    clean_control_status=np.where(values[:,:6] >= 0.05, 'PASS', 'FAIL')
    clean_g2=np.array([conjunctive_status(clean_control_status[i,:]) for i in range(len(env_ids))])

    # G4: two other mandatory requirements with explicit item-level statuses.
    clean_req_status=np.column_stack([np.where(values[:,16]>=0.15,'PASS','FAIL'),
                                      np.where(values[:,17]>=0.15,'PASS','FAIL')])
    clean_g4=np.array([conjunctive_status(clean_req_status[i,:]) for i in range(len(env_ids))])

    clean_g3=np.where((latency_base < 200.0) & (memory_base < 100.0), 'PASS', 'FAIL')
    oracle_eligible=np.array([eligible(clean_g1[i],clean_g2[i],clean_g3[i],clean_g4[i]) for i in range(len(env_ids))],dtype=bool)

    # Apply mutually exclusive evidence degradation to passing G2/G4 items only.
    # Known numeric FAIL remains FAIL; missing/stale/unreviewed evidence becomes UNKNOWN.
    rng_evidence=np.random.default_rng(CONFIG['seeds']['evidence_injection'])
    evidence_rows=[]
    observed_control_states=clean_control_status.astype(object).copy()
    observed_req_states=clean_req_status.astype(object).copy()
    for i,eid in enumerate(env_ids):
        for c in range(6):
            clean=str(clean_control_status[i,c]); obs=clean; injected='none'
            if clean == 'PASS':
                u=float(rng_evidence.random())
                if u < 0.05: obs='UNKNOWN'; injected='missing'
                elif u < 0.10: obs='UNKNOWN'; injected='stale'
                elif u < 0.15: obs='UNKNOWN'; injected='unreviewed'
            observed_control_states[i,c]=obs
            evidence_rows.append({'environment_id':eid,'gate_id':'G2','item_id':f'CTRL{c+1:02d}',
                                  'clean_state':clean,'observed_state':obs,'injection':injected})
        for j,req in enumerate(['REQ_A','REQ_B']):
            clean=str(clean_req_status[i,j]); obs=clean; injected='none'
            if clean == 'PASS':
                u=float(rng_evidence.random())
                if u < 0.05: obs='UNKNOWN'; injected='missing'
                elif u < 0.10: obs='UNKNOWN'; injected='stale'
                elif u < 0.15: obs='UNKNOWN'; injected='unreviewed'
            observed_req_states[i,j]=obs
            evidence_rows.append({'environment_id':eid,'gate_id':'G4','item_id':req,
                                  'clean_state':clean,'observed_state':obs,'injection':injected})

    # Thirty noisy latency and memory readings per environment. Separate RNG, same declared seed.
    rng_noise=np.random.default_rng(CONFIG['seeds']['measurement_noise'])
    sigma=CONFIG['measurement_noise_lognormal_sigma']
    execution_rows=[]; resource_rows=[]
    for i,eid in enumerate(env_ids):
        lat_mult=rng_noise.lognormal(mean=-0.5*sigma*sigma, sigma=sigma, size=30)
        mem_mult=rng_noise.lognormal(mean=-0.5*sigma*sigma, sigma=sigma, size=30)
        lats=latency_base[i]*lat_mult
        mems=memory_base[i]*mem_mult
        for rep in range(30):
            execution_rows.append({'environment_id':eid,'repetition':rep+1,
                'latency_ms':float(lats[rep]),'memory_mb':float(mems[rep]),
                'latency_exceedance':bool(lats[rep] >= 200.0),'memory_exceedance':bool(mems[rep] >= 100.0),
                'any_exceedance':bool(lats[rep] >= 200.0 or mems[rep] >= 100.0)})
        resource_rows.append({'environment_id':eid,'latency_baseline_ms':float(latency_base[i]),
            'memory_baseline_mb':float(memory_base[i]),'latency_median_ms':float(np.median(lats)),
            'latency_p95_ms':float(np.percentile(lats,95)),'latency_p99_ms':float(np.percentile(lats,99)),
            'memory_peak_mb':float(np.max(mems)),
            'latency_exceedance_count':int(np.sum(lats>=200.0)),
            'memory_exceedance_count':int(np.sum(mems>=100.0)),
            'any_exceedance_count':int(np.sum((lats>=200.0)|(mems>=100.0)))})
    exdf=pd.DataFrame(execution_rows); resdf=pd.DataFrame(resource_rows)
    csv_write(output_dir/'execution_log.csv', exdf)
    csv_write(output_dir/'evidence_states.csv', evidence_rows)

    resource_stats=resdf.set_index('environment_id')
    observed_g1=clean_g1.copy()
    observed_g2=np.array([conjunctive_status(observed_control_states[i,:]) for i in range(len(env_ids))])
    observed_g3=np.where((resdf['latency_p95_ms'].to_numpy()<200.0)&(resdf['memory_peak_mb'].to_numpy()<100.0),'PASS','FAIL')
    observed_g4=np.array([conjunctive_status(observed_req_states[i,:]) for i in range(len(env_ids))])
    observed_eligible=np.array([eligible(observed_g1[i],observed_g2[i],observed_g3[i],observed_g4[i]) for i in range(len(env_ids))],dtype=bool)
    rows=[]
    for i,eid in enumerate(env_ids):
        sr=scope_records[i]
        row={'environment_id':eid,'dwell_days':float(dwell[i]),'latency_baseline_ms':float(latency_base[i]),
             'memory_baseline_mb':float(memory_base[i]),'G1_clean':clean_g1[i], 'G2_clean':clean_g2[i],
             'G3_clean':clean_g3[i], 'G4_clean':clean_g4[i], 'oracle_eligible':bool(oracle_eligible[i]),
             'G1_observed':observed_g1[i], 'G2_observed':observed_g2[i], 'G3_observed':observed_g3[i],
             'G4_observed':observed_g4[i], 'observed_eligible':bool(observed_eligible[i]),
             'scope_system_boundary_defined':sr['system_boundary_defined'],
             'scope_jurisdiction_resolved':sr['jurisdiction_resolved'],
             'scope_control_registry_version_present':sr['control_registry_version_present'],
             'scope_applicability_rationale_present':sr['applicability_rationale_present'],
             'scope_known_conflict':sr['known_scope_conflict']}
        for c in range(6):
            row[f'CTRL{c+1:02d}_clean_state']=str(clean_control_status[i,c])
            row[f'CTRL{c+1:02d}_observed_state']=str(observed_control_states[i,c])
        for j,req in enumerate(['REQ_A','REQ_B']):
            row[f'{req}_clean_state']=str(clean_req_status[i,j])
            row[f'{req}_observed_state']=str(observed_req_states[i,j])
        row.update({INDICATORS[j]:float(values[i,j]) for j in range(20)})
        row.update({f'D{d+1}_rating':float(D[i,d]*5) for d in range(6)})
        row.update(resource_stats.loc[eid].to_dict())
        rows.append(row)
    corpus=pd.DataFrame(rows)

    # Repeated stratified 5-fold CV. Ten deterministic repeats describe split variability.
    # All model thresholds are fixed to 0.50 before evaluation; no test fold selects any setting.
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=observed_eligible.astype(int)
    class_counts=np.bincount(y, minlength=2)
    if len(np.unique(y)) < 2 or np.min(class_counts) < 5:
        raise RuntimeError(f"Too few cases in one class for stratified 5-fold CV: class counts={class_counts.tolist()}")
    corpus['split']='repeated_stratified_5fold_cv'
    # Encode every categorical gate input explicitly for the full-input surrogate.
    full_cols=[]
    categorical_inputs=['G1_observed'] + [f'CTRL{c+1:02d}_observed_state' for c in range(6)] + ['REQ_A_observed_state','REQ_B_observed_state']
    for field in categorical_inputs:
        for state in ['PASS','FAIL','UNKNOWN']:
            col=f'{field}_{state.lower()}'
            corpus[col]=(corpus[field]==state).astype(int)
            full_cols.append(col)
    full_cols += ['latency_p95_ms','memory_peak_mb']
    dom_cols=[f'D{d+1}_rating' for d in range(6)]
    model_names=['oracle','logistic_regression_full','logistic_regression_6domain','equal_weight']
    pred_rows=[]; repeat_metrics=[]
    for repeat in range(10):
        oof={name:np.full(len(y), np.nan, dtype=float) for name in model_names}
        splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['seeds']['cv_fold_seed']+repeat)
        for fold, (train_ix, test_ix) in enumerate(splitter.split(corpus, y), start=1):
            for name, cols in [('logistic_regression_full', full_cols), ('logistic_regression_6domain', dom_cols)]:
                pipe=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=101))])
                pipe.fit(corpus.iloc[train_ix][cols].astype(float), y[train_ix])
                oof[name][test_ix]=pipe.predict_proba(corpus.iloc[test_ix][cols].astype(float))[:,1]
            oof['oracle'][test_ix]=y[test_ix].astype(float)
            oof['equal_weight'][test_ix]=corpus.iloc[test_ix][dom_cols].mean(axis=1).to_numpy()/5.0
        if any(np.isnan(oof[name]).any() for name in model_names):
            raise RuntimeError(f"Incomplete out-of-fold predictions in repeat {repeat+1}")
        by_model={}
        for name in model_names:
            score=oof[name]
            pred=(score>=0.50).astype(int)
            m=safe_metrics(y,pred)
            m['auroc']=float(roc_auc_score(y,score))
            m['average_precision']=float(average_precision_score(y,score))
            m.update({'repeat':repeat+1,'model':name,'threshold':0.50,'folds':5,
                      'evaluation_cases':len(y),'positive_support':int(y.sum()),'negative_support':int((1-y).sum())})
            repeat_metrics.append(m)
            by_model[name]=(score,pred)
        for i,eid in enumerate(env_ids):
            rec={'repeat':repeat+1,'environment_id':eid,'y_true':int(y[i])}
            for name in model_names:
                rec[f'{name}_score']=float(by_model[name][0][i])
                rec[f'{name}_prediction']=int(by_model[name][1][i])
            pred_rows.append(rec)
    pred_df=pd.DataFrame(pred_rows)
    metrics_df=pd.DataFrame(repeat_metrics)
    pred_df.to_csv(output_dir/'cv_oof_predictions.csv', index=False, lineterminator='\n', float_format='%.8f')
    csv_write(output_dir/'cv_metrics_by_repeat.csv', metrics_df)
    metric_cols=['balanced_accuracy','sensitivity','specificity','precision_positive','macro_f1','accuracy','auroc','average_precision']
    summaries=[]
    for name in model_names:
        subset=metrics_df[metrics_df.model==name]
        item={'model':name,'repeats':int(subset.repeat.nunique()),'folds_per_repeat':5,
              'n_environments':len(y),'positive_support':int(y.sum()),'negative_support':int((1-y).sum()),'threshold':0.50,
              'interpretation':('tautological specification-identity check, not predictive performance' if name=='oracle' else 'descriptive repeated-CV rule-recovery performance')}
        for col in metric_cols:
            vals=subset[col].astype(float)
            item[f'{col}_mean']=float(vals.mean())
            item[f'{col}_sd_across_repeats']=float(vals.std(ddof=1))
            item[f'{col}_min']=float(vals.min())
            item[f'{col}_max']=float(vals.max())
        summaries.append(item)
    csv_write(output_dir/'cv_metrics_summary.csv', summaries)

    # Compact decision-record table; model evaluation uses repeated stratified 5-fold cross-validation.
    decision_cols=['environment_id','G1_clean','G2_clean','G3_clean','G4_clean','oracle_eligible',
                   'G1_observed','G2_observed','G3_observed','G4_observed','observed_eligible',
                   'scope_system_boundary_defined','scope_jurisdiction_resolved','scope_control_registry_version_present',
                   'scope_applicability_rationale_present','scope_known_conflict',
                   *[f'CTRL{c+1:02d}_observed_state' for c in range(6)],'REQ_A_observed_state','REQ_B_observed_state',
                   'latency_median_ms','latency_p95_ms','latency_p99_ms','memory_peak_mb',
                   'latency_exceedance_count','memory_exceedance_count','any_exceedance_count']
    csv_write(output_dir/'decision_records.csv', corpus[decision_cols])

    # Threshold sensitivity on the same environment records; only G3 thresholds change.
    threshold_rows=[]
    for lt in [150,175,200,250,300]:
        for mt in [64,100,150,256]:
            pass_g3=(resdf['latency_p95_ms'].to_numpy()<lt)&(resdf['memory_peak_mb'].to_numpy()<mt)
            count=sum((observed_g1=='PASS')&(observed_g2=='PASS')&pass_g3&(observed_g4=='PASS'))
            threshold_rows.append({'latency_p95_limit_ms_strict':lt,'peak_memory_limit_mb_strict':mt,
                                   'eligible_n':int(count),'eligible_fraction':float(count/len(env_ids))})
    csv_write(output_dir/'threshold_sensitivity.csv', threshold_rows)

    # Additional evidence-state injection sensitivity, layered on the base schedule across six G2 controls + two G4 requirements.
    base_evidence=pd.DataFrame(evidence_rows)
    rrng=np.random.default_rng(99+500)
    extra_u=rrng.random((len(env_ids),8))
    sens_rows=[]
    for rate in [0.0,0.05,0.10,0.20]:
        g2_states=[]; g4_states=[]
        for i,eid in enumerate(env_ids):
            rows_i=base_evidence[base_evidence.environment_id==eid]
            ctrl=rows_i[rows_i.gate_id=='G2'].sort_values('item_id')['observed_state'].tolist()
            req=rows_i[rows_i.gate_id=='G4'].sort_values('item_id')['observed_state'].tolist()
            for c,state in enumerate(ctrl):
                if state=='PASS' and extra_u[i,c] < rate: ctrl[c]='UNKNOWN'
            for j,state in enumerate(req):
                if state=='PASS' and extra_u[i,6+j] < rate: req[j]='UNKNOWN'
            g2_states.append(conjunctive_status(ctrl)); g4_states.append(conjunctive_status(req))
        g2_states=np.asarray(g2_states); g4_states=np.asarray(g4_states)
        count=int(np.sum((observed_g1=='PASS')&(g2_states=='PASS')&(observed_g3=='PASS')&(g4_states=='PASS')))
        sens_rows.append({'additional_degradation_probability_per_pass_item':rate,
                          'eligible_n':count,'eligible_fraction':count/len(env_ids),
                          'g2_pass_n':int(np.sum(g2_states=='PASS')),'g2_fail_n':int(np.sum(g2_states=='FAIL')),
                          'g2_unknown_n':int(np.sum(g2_states=='UNKNOWN')),
                          'g4_pass_n':int(np.sum(g4_states=='PASS')),'g4_fail_n':int(np.sum(g4_states=='FAIL')),
                          'g4_unknown_n':int(np.sum(g4_states=='UNKNOWN')),
                          'unknown_to_eligible_conversions':0})
    csv_write(output_dir/'evidence_injection_sensitivity.csv', sens_rows)

    # Post-gate two-resource dominance: one unordered pair counted if either candidate dominates the other.
    eligible_ids=[env_ids[i] for i in range(len(env_ids)) if observed_eligible[i]]
    cand=resdf[resdf.environment_id.isin(eligible_ids)].sort_values('environment_id').to_dict('records')
    dominance=0
    for i in range(len(cand)):
        for j in range(i+1,len(cand)):
            a,b=cand[i],cand[j]
            a_dom=(a['latency_median_ms']<=b['latency_median_ms'] and a['memory_peak_mb']<=b['memory_peak_mb'] and
                   (a['latency_median_ms']<b['latency_median_ms'] or a['memory_peak_mb']<b['memory_peak_mb']))
            b_dom=(b['latency_median_ms']<=a['latency_median_ms'] and b['memory_peak_mb']<=a['memory_peak_mb'] and
                   (b['latency_median_ms']<a['latency_median_ms'] or b['memory_peak_mb']<a['memory_peak_mb']))
            if a_dom or b_dom: dominance += 1
    csv_write(output_dir/'postgate_resources.csv', pd.DataFrame(cand) if cand else pd.DataFrame(columns=resdf.columns))
    summary={
      'run_id':CONFIG['run_id'], 'n_environments':len(env_ids), 'n_execution_rows':len(execution_rows),
      'oracle_eligible_n':int(oracle_eligible.sum()),'observed_eligible_n':int(observed_eligible.sum()),
      'oracle_observed_agreement_n':int(np.sum(oracle_eligible==observed_eligible)),
      'oracle_observed_agreement_fraction':float(np.mean(oracle_eligible==observed_eligible)),
      'oracle_only_positive_n':int(np.sum(oracle_eligible & ~observed_eligible)),
      'observed_only_positive_n':int(np.sum(~oracle_eligible & observed_eligible)),
      'gate_state_counts':{g:{state:int(np.sum(arr==state)) for state in ['PASS','FAIL','UNKNOWN']} for g,arr in [('G1',observed_g1),('G2',observed_g2),('G3',observed_g3),('G4',observed_g4)]},
      'scope_scenario_rates':{'PASS':0.90,'FAIL':0.05,'UNKNOWN':0.05,'basis':'analyst-selected synthetic scenario, not empirical rates'},
      'g1_pass_n':int(np.sum(observed_g1=='PASS')),'g2_pass_n':int(np.sum(observed_g2=='PASS')),
      'g3_pass_n':int(np.sum(observed_g3=='PASS')),'g4_pass_n':int(np.sum(observed_g4=='PASS')), 
      'all_execution_pass_env_n':int(np.sum(resdf['any_exceedance_count'].to_numpy()==0)),
      'partial_exceedance_env_n':int(np.sum((resdf['any_exceedance_count'].to_numpy()>0)&(resdf['any_exceedance_count'].to_numpy()<30))),
      'full_exceedance_env_n':int(np.sum(resdf['any_exceedance_count'].to_numpy()==30)),
      'latency_exceedances_n':int(exdf['latency_exceedance'].sum()),'memory_exceedances_n':int(exdf['memory_exceedance'].sum()),
      'any_exceedance_n':int(exdf['any_exceedance'].sum()),'both_exceedance_n':int((exdf['latency_exceedance']&exdf['memory_exceedance']).sum()),
      'latency_exceedance_fraction':float(exdf['latency_exceedance'].mean()),
      'memory_exceedance_fraction':float(exdf['memory_exceedance'].mean()),
      'any_exceedance_fraction':float(exdf['any_exceedance'].mean()),
      'dominance_unordered_pairs':int(dominance),'postgate_candidate_n':len(cand),
      'correlation_min_eigenvalue':float(eig.min()),
      'class_counts':{'observed_negative':int(np.sum(y==0)),'observed_positive':int(np.sum(y==1))},
      'cross_validation':{'repeats':10,'folds_per_repeat':5,'threshold':0.50,'positive_n':int(np.sum(y==1)),'negative_n':int(np.sum(y==0))},
      'interpretation': 'Candidate new-run outputs generated from newly declared factor model and rules; not a reproduction of the 19 October locked run.'
    }
    json_dump(output_dir/'summary_results.json', summary)

    # Core logic tests for the exact conjunctive predicate and strict resource boundary.
    tests=[]
    def check(name, condition, detail=''):
        tests.append({'test':name,'passed':bool(condition),'detail':detail})
    check('all four PASS is eligible',eligible('PASS','PASS','PASS','PASS'))
    check('numeric G1 non-passing value is FAIL', (0.14 < 0.15) and not eligible('FAIL','PASS','PASS','PASS'))
    check('numeric G4 non-passing value is FAIL', (0.14 < 0.15) and not eligible('PASS','PASS','PASS','FAIL'))
    check('numeric subcontrol failure contributes FAIL', conjunctive_status(['PASS','FAIL','PASS'])=='FAIL')
    check('FAIL dominates UNKNOWN in conjunction', conjunctive_status(['PASS','UNKNOWN','FAIL'])=='FAIL')
    check('UNKNOWN used when unresolved and no FAIL', conjunctive_status(['PASS','UNKNOWN','PASS'])=='UNKNOWN')
    check('G1 UNKNOWN blocks',not eligible('UNKNOWN','PASS','PASS','PASS'))
    check('G2 UNKNOWN blocks',not eligible('PASS','UNKNOWN','PASS','PASS'))
    check('G3 FAIL blocks',not eligible('PASS','PASS','FAIL','PASS'))
    check('G4 UNKNOWN blocks',not eligible('PASS','PASS','PASS','UNKNOWN'))
    check('strict latency 199.99 passes threshold',199.99 < 200.0)
    check('strict latency 200.00 fails threshold',not (200.00 < 200.0))
    check('strict memory 99.99 passes threshold',99.99 < 100.0)
    check('strict memory 100.00 fails threshold',not (100.00 < 100.0))
    check('correlation matrix symmetric',np.allclose(corr,corr.T,atol=1e-12))
    check('correlation matrix positive semidefinite',bool(eig.min() >= -1e-10),f'min eigenvalue={eig.min():.8f}')
    check('240 environments',len(corpus)==240)
    check('7200 execution rows',len(exdf)==7200)
    check('G1 scope status has PASS/FAIL/UNKNOWN only',bool(np.isin(observed_g1,['PASS','FAIL','UNKNOWN']).all()))
    check('G1 PASS has complete scope record',bool(((corpus.loc[corpus.G1_observed=='PASS',['scope_system_boundary_defined','scope_jurisdiction_resolved','scope_control_registry_version_present','scope_applicability_rationale_present']]).all(axis=1)).all()))
    check('G1 FAIL has recorded scope conflict',bool(corpus.loc[corpus.G1_observed=='FAIL','scope_known_conflict'].all()))
    check('G1 UNKNOWN has at least one incomplete scope field',bool((~corpus.loc[corpus.G1_observed=='UNKNOWN',['scope_system_boundary_defined','scope_jurisdiction_resolved','scope_control_registry_version_present','scope_applicability_rationale_present']].all(axis=1)).all()))
    check('G4 matches three-valued conjunction of REQ-A and REQ-B',all(observed_g4[i]==conjunctive_status(observed_req_states[i,:]) for i in range(len(env_ids))))
    check('G2 matches three-valued conjunction of six controls',all(observed_g2[i]==conjunctive_status(observed_control_states[i,:]) for i in range(len(env_ids))))
    check('evidence state table contains eight items per environment',len(evidence_rows)==len(env_ids)*8)
    # Regression checks added after independent audit found U4 truncation of UNKNOWN.
    item_columns=[f'CTRL{c+1:02d}_observed_state' for c in range(6)] + ['REQ_A_observed_state','REQ_B_observed_state']
    canonical={'PASS','FAIL','UNKNOWN'}
    evidence_lookup={(r['environment_id'],r['item_id']):r['observed_state'] for r in evidence_rows}
    check('canonical G2/G4 item-state strings (catches UNKN truncation)',
          all(str(v) in canonical for col in item_columns for v in corpus[col].tolist()))
    check('decision item states match independent evidence rows',
          all(str(corpus.at[i,item+'_observed_state']) == evidence_lookup[(eid,item)]
              for i,eid in enumerate(env_ids) for item in [*[f'CTRL{j+1:02d}' for j in range(6)],'REQ_A','REQ_B']))
    check('injected UNKNOWN evidence survives serialization',
          sum(str(corpus.at[i,item+'_observed_state'])=='UNKNOWN'
              for i,eid in enumerate(env_ids) for item in [*[f'CTRL{j+1:02d}' for j in range(6)],'REQ_A','REQ_B']) ==
          sum(r['observed_state']=='UNKNOWN' for r in evidence_rows))
    check('one-hot categorical input has exactly one active state per case',
          all((corpus[[f'{field}_{state.lower()}' for state in ['PASS','FAIL','UNKNOWN']]].sum(axis=1)==1).all()
              for field in categorical_inputs))
    check('numeric G2 control failures remain FAIL',all((clean_control_status[i,c]=='FAIL' and observed_control_states[i,c]=='FAIL') for i in range(len(env_ids)) for c in range(6) if clean_control_status[i,c]=='FAIL'))
    check('numeric G4 requirement failures remain FAIL',all((clean_req_status[i,j]=='FAIL' and observed_req_states[i,j]=='FAIL') for i in range(len(env_ids)) for j in range(2) if clean_req_status[i,j]=='FAIL'))
    check('2400 out-of-fold prediction rows',len(pred_df)==2400)
    check('10 repeats and four model summaries',set(metrics_df['repeat'])==set(range(1,11)) and len(summaries)==4)
    check('all eligible environments pass all four gates',bool(((corpus.loc[corpus.observed_eligible,'G1_observed']=='PASS') &
        (corpus.loc[corpus.observed_eligible,'G2_observed']=='PASS') & (corpus.loc[corpus.observed_eligible,'G3_observed']=='PASS') &
        (corpus.loc[corpus.observed_eligible,'G4_observed']=='PASS')).all()))
    check('20 T1 threshold combinations',len(threshold_rows)==20)
    check('G1-G4 state counts each sum to 240',all(sum(v.values())==240 for v in ({g:{st:int(np.sum(arr==st)) for st in ['PASS','FAIL','UNKNOWN']} for g,arr in [('G1',observed_g1),('G2',observed_g2),('G3',observed_g3),('G4',observed_g4)]}).values()))
    check('T2 injection never increases eligibility',all(sens_rows[i]['eligible_n'] >= sens_rows[i+1]['eligible_n'] for i in range(len(sens_rows)-1)),
          'Monotonicity is checked on this fixed base scenario and the deterministic additional-degradation procedure.')
    csv_write(output_dir/'verification_checks.csv', tests)
    failed=[t for t in tests if not t['passed']]
    (output_dir/'verification.txt').write_text(
        f"Passed {len(tests)-len(failed)}/{len(tests)} checks\n" +
        ''.join(f"{'PASS' if t['passed'] else 'FAIL'} | {t['test']} | {t['detail']}\n" for t in tests),
        encoding='utf-8', newline='\n')
    if failed:
        raise RuntimeError(f"Verification failed: {[t['test'] for t in failed]}")

    # Run provenance. File list is hashed after all outputs except manifest itself.
    files={p.name:sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name!='manifest.json'}
    manifest={
      'run_id':CONFIG['run_id'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'provenance_status':'new, explicitly specified candidate simulation; not the locked 19 October run',
      'locked_rule_set_hash_recovered':False,
      'parent_26_october_bundle_used_as_input':False,
      'generated_by':'phase1_newrun.py', 'generator_sha256':sha256_file(Path(__file__).resolve()),
      'config':CONFIG,
      'runtime':{'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__,
                 'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
      'correlation':{'min_eigenvalue':float(eig.min()),'max_eigenvalue':float(eig.max()),
                     'positive_semidefinite':bool(eig.min()>=-1e-10)},
      'summary_results':summary,'verification_passed':len(tests)-len(failed),'verification_total':len(tests),
      'metric_definitions':{'balanced_accuracy':'(sensitivity + specificity) / 2',
        'macro_f1':'arithmetic mean of positive-class and negative-class F1',
        'auroc':'area under receiver operating characteristic curve using out-of-fold scores',
        'average_precision':'precision-recall average precision using out-of-fold scores',
        'repeated_cv_summary':'mean, sample SD, min, and max over 10 repeated stratified 5-fold evaluations; split variability, not a confidence interval',
        'oracle':'observed labels re-emitted as predictions; tautological identity check, not an independent comparator'},
      'sha256_files':files,
      'limitations':['The 20x20 latent Gaussian-copula correlation matrix is newly specified by a factor model, not the manuscript-cited empirical matrix.',
        'Scope-state distribution, indicator-to-control mapping, REQ-A/REQ-B thresholds, and synthetic latency/memory equations are newly specified assumptions; conclusions are limited to this specification.',
        'All resource measurements are simulated, not Raspberry Pi 4 measurements.',
        'No Phase II NHS pilot data or real-world gate validation are present; the pilot is uncompleted.',
        'The generator does not ingest CTI events or implement the full CTI-to-evidence architecture; it evaluates a synthetic four-gate decision function under declared assumptions.',
        'The 10 repeated CV runs reuse the same 240 synthetic environments; repeat-to-repeat SD describes split variability, not independent sampling uncertainty.',
        'This package does not claim equality with or reconstruction of the 19 October 2026 run.']
    }
    json_dump(output_dir/'manifest.json', manifest)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,help='new output directory; must not exist or must be empty')
    args=ap.parse_args()
    out=Path(args.out).expanduser().resolve()
    summary=run(out)
    print(f"Generated: {out}")
    print(f"Environments: {summary['n_environments']} | executions: {summary['n_execution_rows']}")
    print(f"Observed eligible: {summary['observed_eligible_n']} | clean/oracle eligible: {summary['oracle_eligible_n']}")
    print(f"Clean-oracle/observed agreement: {summary['oracle_observed_agreement_n']}/240 ({summary['oracle_observed_agreement_fraction']:.2%})")
    print(f"Resource exceedances: latency={summary['latency_exceedances_n']}, memory={summary['memory_exceedances_n']}, any={summary['any_exceedance_n']}")
    print((out/'verification.txt').read_text(encoding='utf-8'))
    print('Repeated CV summary:')
    print((out/'cv_metrics_summary.csv').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
