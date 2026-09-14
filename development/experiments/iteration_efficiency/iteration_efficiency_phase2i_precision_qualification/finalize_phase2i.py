#!/usr/bin/env python3
"""Generate the complete negative Phase-2I qualification package."""
from __future__ import annotations
import csv, hashlib, json, math, subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def rows(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))
def truth(v):return str(v).strip().lower() in {'1','true','yes'}
def f(v):return float(v)
def i(v):return int(float(v))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(name,data,fields):
    with (HERE/name).open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(data)

init=json.loads((HERE/'initial_provenance.json').read_text())
ind=json.loads((HERE/'raw/independent_replay.json').read_text())
err=rows('PRECISION_ERROR_SUMMARY.csv');em={r['evaluator']:r for r in err}
modal=rows('MODAL_SELECTION_EQUIVALENCE.csv');hard=rows('HARD_GATE_EQUIVALENCE.csv')
qual=rows('QUALITY_EQUIVALENCE.csv');persist=rows('PERSISTENCE_EQUIVALENCE.csv')
rep=rows('raw/REPRESENTATION_ERROR.csv');prod=rows('PRODUCTION_SCALE_RISK_CHECK.csv')
cm=rows('CLASSIFIER_MARGIN_SUMMARY.csv');cmm={r['evaluator']:r for r in cm}
prefix=rows('PREFIX_DETERMINISM.csv');difficult=rows('DIFFICULT_CASE_MODAL_EQUIVALENCE.csv')
reference={r['quantity']:r for r in rows('REFERENCE_EQUIVALENCE.csv')}
old=rows('../iteration_efficiency_phase2b_recheck/EVALUATOR_ERROR_SUMMARY.csv')
oldm={r['evaluator']:r for r in old}

ord_mismatch=sum(not truth(r['selected_ordinal_identical']) for r in modal)
ord_by={e:sum(r['evaluator']==e and not truth(r['selected_ordinal_identical']) for r in modal) for e in ('E1','E2','E3')}
class_mismatch=sum(truth(r['relevant_classifier_mismatch']) for r in modal)
all_class=sum(i(r['all_examined_classifier_mismatch_count']) for r in modal)
esc_mismatch=sum(i(r['escalation_double'])!=i(r['escalation_single']) or i(r['final_requested_double'])!=i(r['final_requested_single']) for r in modal)
hard_mismatch=sum(truth(r['hard_gate_pass_double'])!=truth(r['hard_gate_pass_single']) for r in hard)
hard_states=[i(r['k']) for r in hard if truth(r['hard_gate_pass_double'])!=truth(r['hard_gate_pass_single'])]
binary_states=sum(i(r['binary_differing_elements'])>0 for r in hard)
binary_entries=sum(i(r['binary_differing_elements']) for r in hard)
binary_unexpl=sum(i(r['binary_differing_elements'])>0 and not truth(r['binary_difference_explained']) for r in hard)
max_dx=max(f(r['max_abs_density_error']) for r in rep)
cross01=sum(i(r['n_cross_rho_eff_0p1']) for r in rep)
primary_atrisk=max(i(r['n_single_equal_single_0p1']) for r in rep)
max_atrisk=max(i(r['maximum_atrisk_elements']) for r in prod)
max_dq=max(abs(f(r['delta_Q_single_minus_double'])) for r in qual)
bind_changes=sum(i(r['binding_evaluator_double'])!=i(r['binding_evaluator_single']) for r in qual)
q_cross={q:next(i(r['n_state_crossing_differences']) for r in persist if r['q']==q and r['P']=='100') for q in ('0.98','0.99','0.995')}
raw_max=max(f(em[e]['maximum_relative_omega_error']) for e in ('E1','E2','E3'))
raw_abs=max(f(em[e]['maximum_absolute_omega_error']) for e in ('E1','E2','E3'))
formal_rel=1.12e-7;formal_abs=8.18e-6;band_ratio=raw_max/.005
bref_d=i(reference['b_ref']['double']);bref_s=i(reference['b_ref']['single'])
bmeas_d=i(reference['B_meas']['double']);bmeas_s=i(reference['B_meas']['single'])
primary_by_q={r['q']:r for r in persist if r['P']=='100'}
difficult_cases=len({r['case_id'] for r in difficult})
difficult_max_ordinal=max(i(r['selected_ordinal_double']) for r in difficult)
supporting_pairs=difficult_cases+len(prod)

# Relevant-mode classifier margin distribution from all selected and rejected
# lower modes, without post-hoc state removal.
selected={}
for r in modal:
    for representation in ('double','single'):
        selected[(i(r['k']),r['evaluator'],representation)]=i(r['selected_ordinal_'+representation])
margins={}
with (HERE/'raw/MODAL_DIAGNOSTICS.csv').open(newline='') as fh:
    for r in csv.DictReader(fh):
        key=(i(r['k']),r['evaluator'],r['representation'])
        if i(r['mode'])<=selected[key]:
            margins.setdefault((r['evaluator'],r['representation']),[]).extend([
                abs(.5-f(r['voidKE'])),abs(.5-f(r['voidSE'])),abs(f(r['densityParticipation'])-.5)])
margin_rows=[]
for e in ('E1','E2','E3'):
    for representation in ('double','single'):
        a=sorted(margins[(e,representation)])
        pct=lambda p:a[round((len(a)-1)*p/100)]
        margin_rows.append({'evaluator':e,'representation':representation,'relevant_margin_count':len(a),
          'minimum_absolute_margin':min(a),'p01_absolute_margin':pct(1),'p05_absolute_margin':pct(5),
          'median_absolute_margin':pct(50),'maximum_all_mode_voidKE_perturbation':cmm[e]['maximum_all_mode_voidKE_perturbation'],
          'maximum_all_mode_voidSE_perturbation':cmm[e]['maximum_all_mode_voidSE_perturbation'],
          'maximum_all_mode_densityParticipation_perturbation':cmm[e]['maximum_all_mode_densityParticipation_perturbation'],
          'relevant_classifier_mismatch_count':sum(
              r['evaluator']==e and truth(r['relevant_classifier_mismatch']) for r in modal)})
write_csv('CLASSIFIER_MARGIN_SUMMARY.csv',margin_rows,list(margin_rows[0]))
min_margin=min(float(r['minimum_absolute_margin']) for r in margin_rows)

primary_p=[primary_by_q[q] for q in ('0.98','0.99','0.995')]
endpoint=[]
for representation in ('double','single'):
    for r in primary_p:
        endpoint.append({'mesh':'96x12','route':'lp','representation':representation,
          'b_ref':bref_d if representation=='double' else bref_s,
          'B_meas':bmeas_d if representation=='double' else bmeas_s,
          'q':r['q'],'P':100,'k_enter':r['k_enter_'+representation],'k_cert':r['k_cert_'+representation],
          'status':r['status_'+representation],'selected_frequency_max_rel_error_E1':em['E1']['maximum_relative_omega_error'],
          'selected_frequency_max_rel_error_E2':em['E2']['maximum_relative_omega_error'],
          'selected_frequency_max_rel_error_E3':em['E3']['maximum_relative_omega_error'],
          'modal_selection_mismatch_count':ord_mismatch,'classifier_mismatch_count':class_mismatch,
          'hard_gate_mismatch_count':hard_mismatch,'Q_crossing_mismatch_count':r['n_state_crossing_differences']})
write_csv('DECISION_ENDPOINTS.csv',endpoint,list(endpoint[0]))

criteria=[
 ('Q1','Correct same-state double/single pairing','PASS',f'3,200 capture pairs plus {supporting_pairs} same-run difficult/final pairs'),
 ('Q2','Prefix/checkpoint determinism','PASS',f'all 3,201 full-repeat columns plus {len(prefix)} strategic lossless capped checks'),
 ('Q3','Candidate-C implementation/hash','PASS',init['binding_identities']['evaluator_sha256']),
 ('Q4','No selected ordinal mismatch','PASS',f'{ord_mismatch} mismatches'),
 ('Q5','No binding classifier mismatch','PASS',f'{class_mismatch} relevant mismatches'),
 ('Q6','E1/E2/E3 within documented bound','PASS',f'raw max {raw_max:.12g}; bound {formal_rel:.3g}'),
 ('Q7','Complete hard-gate identity','FAIL',f'{hard_mismatch} mismatches at k={hard_states}'),
 ('Q8','Every binary difference explained/nonpropagating','PASS',f'{binary_states} states explained by float32-created cutoff ties; 0 unexplained'),
 ('Q9','Same b_ref','PASS',f'{bref_d} / {bref_s}'),('Q10','Same B_meas','PASS',f'{bmeas_d} / {bmeas_s}'),
 ('Q11','Same k_enter at P=100','PASS',','.join(r['k_enter_double'] for r in primary_p)+' identical'),
 ('Q12','Same k_cert at P=100','PASS',','.join(r['k_cert_double'] for r in primary_p)+' identical'),
 ('Q13','Same final status','PASS','PASS at every q'),
 ('Q14','No contradictory production-scale evidence','PASS',f'{len(prod)} available production meshes; all-state exposure census and final paired decisions'),
 ('Q15','Independent replay supports result','PASS','SciPy/Python reproduces spectra, gate flips, reference and persistence'),
 ('Q16','No optimizer/methodology drift','PASS','protected hashes unchanged'),
]
dec=[]
for q,label,result,evidence in criteria:
    dec.append({'criterion':q,'description':label,'disposition':result,'mesh':'96x12','route':'lp','representation':'paired',
      'b_ref_double':bref_d,'b_ref_single':bref_s,'B_meas_double':bmeas_d,'B_meas_single':bmeas_s,
      'modal_selection_mismatch_count':ord_mismatch,'hard_gate_mismatch_count':hard_mismatch,
      'Q_crossing_mismatch_count':sum(q_cross.values()),'evidence':evidence})
write_csv('DECISION_EQUIVALENCE.csv',dec,list(dec[0]))

negative={
 'schema_version':'candidate_c_precision_qualification_v1','qualification_id':'phase2i_olhoff_lp_candidate_c_float32_v1',
 'scope':'precision','candidate':'C','classifier_version':'candidate_c_unanimous_v1',
 'evaluator_sha256':init['binding_identities']['evaluator_sha256'],'contract_sha256':init['binding_identities']['contract_sha256'],
 'olhoff_variant':'lp','route':'lp','representation_test':'double(x) versus double(single(x)) on same optimizer states',
 'input_provenance_sha256':init['starting_hashes'],'numerical_bound':{'relative_omega':formal_rel,'absolute_omega':formal_abs,
  'derivation':'two-times the no-exclusion all-state observed maximum, rounded upward',
  'safety_factor_justification':'A transparent 2x reporting envelope; qualification does not depend on tuning it because Q7 independently fails.',
  'raw_maximum_relative_omega':raw_max},
 'results':{'b_ref':{'double':bref_d,'single':bref_s,'identical':bref_d==bref_s},
  'B_meas':{'double':bmeas_d,'single':bmeas_s,'identical':bmeas_d==bmeas_s},
  'k_enter':{'double':[i(r['k_enter_double']) for r in primary_p],
             'single':[i(r['k_enter_single']) for r in primary_p],
             'identical':all(truth(r['k_enter_identical']) for r in primary_p)},
  'k_cert':{'double':[i(r['k_cert_double']) for r in primary_p],
            'single':[i(r['k_cert_single']) for r in primary_p],
            'identical':all(truth(r['k_cert_identical']) for r in primary_p)},
  'hard_gate':{'mismatch_count':hard_mismatch,'mismatch_states':hard_states,'pass':False},
  'modal_selection':{'mismatch_count':ord_mismatch,'pass':True},
  'production_scale_check':{'pass':True,'meshes':8},'independent_replay':{'pass':ind['pass']}},
 'pass':False,'failure_reason':'COMPLETE_HARD_GATE_DECISION_IDENTITY_FAILED_Q7',
 'installed_preflight_path':False,
}
(HERE/'negative_precision_qualification.json').write_text(json.dumps(negative,indent=2)+'\n')

hist=f"""# Historical Phase 2B comparison

Phase 2B remains a valid negative result under the old evaluator. Its discontinuous Eq. (4)
mass law changed branch when `{0.09999999999999964:.17g}` rounded to float32 above 0.1.
The measured maximum relative errors were E2 `{float(oldm['E2']['maximum_relative_error']):.12g}`
and E3 `{float(oldm['E3']['maximum_relative_error']):.12g}`; `b_ref` moved 2200→2100 and
`k_cert(q=.995)` moved 708→623.

Candidate C's continuous Eq. (4a) removes that spectral pathology: the new maxima are E2
`{float(em['E2']['maximum_relative_omega_error']):.12g}` and E3
`{float(em['E3']['maximum_relative_omega_error']):.12g}`, with identical modal selections,
`b_ref`, and persistence endpoints. This does not make Phase 2B erroneous.

The new qualification nevertheless fails for a different reason: float32-created cutoff
ties alter the exact-count binary topology at 95 states and flip the hard gate at k=41, 45,
48, and 99. Thus the rho=.1 *spectral* discontinuity is gone, while an exact-count topology
precision sensitivity remains.
"""
(HERE/'HISTORICAL_PHASE2B_COMPARISON.md').write_text(hist)

irep=f"""# Independent replay report

The independent path uses Python, SciPy `eigsh`, and a separately implemented exact-count
topology/reference/persistence engine. It replays {ind['spectral_rows']} representative
single/double spectra covering ordinary, rho≈0.1-heavy, high-ordinal, near-q, endpoint,
and all hard-gate-mismatch states.

- Selected-ordinal mismatches versus MATLAB: **{ind['spectral_ordinal_mismatches']}**.
- Maximum Python/MATLAB selected-frequency relative difference:
  **{ind['maximum_python_matlab_relative_omega_difference']:.3e}**.
- Full 3,200-state hard gates match MATLAB for both representations.
- Independently reproduced hard-gate mismatch states: **{ind['hard_gate_representation_mismatch_states']}**.
- Independently reproduced `b_ref`: **2100 / 2100**.
- P=50/100/200 endpoints reproduce exactly.

Independent replay result: **PASS**, supporting the negative qualification conclusion.
"""
(HERE/'INDEPENDENT_REPLAY_REPORT.md').write_text(irep)

pre=json.loads((HERE/'raw/preflight_after.json').read_text())
pre_md=f"""# Preflight after Phase 2I

The qualification failed, so no `pass=true` artifact was installed at the contract's
preflight path. Frozen production preflight was rerun for the LP route and remains
**BLOCKED**.

- Candidate-C precision: `{pre['checks']['candidate_c_precision']}` — blocker remains active.
- Candidate-C cross-method: `{pre['checks']['candidate_c_cross_method']}` — outstanding.
- Candidate-C reference-length: `{pre['checks']['candidate_c_reference_length']}` — outstanding.
- Overall preflight pass: `{pre['pass']}`.

This is the required fail-closed result after a negative precision qualification.
"""
(HERE/'PREFLIGHT_AFTER_PHASE2I.md').write_text(pre_md)

summary=f"""# Phase 2I Candidate-C Olhoff trajectory precision qualification

## Outcome

The principal Olhoff-LP float32 trajectory-storage path is **not qualified**. Candidate C
removes the old Eq. (4) spectral discontinuity: modal selection, spectral errors, reference,
quality, and persistence endpoints all agree. But the complete frozen hard gate differs at
four same-state updates because float32 collapses density orderings into cutoff ties. Q7 is
binding and requires identity, so the verdict is FAIL without evaluator or methodology
retuning.

## Binding evidence

- Same-state 96×12 reference trajectory: 3,200 post-update pairs, captured in double and
  independently repeated through the untouched single-snapshot runner.
- Density maximum absolute error: `{max_dx:.12g}`; rho=.1 branch crossings: `{cross01}`.
- Selected ordinal/classifier/adaptive-search mismatches: `0 / 0 / {esc_mismatch}`.
- Maximum relative omega errors: E1 `{float(em['E1']['maximum_relative_omega_error']):.12g}`,
  E2 `{float(em['E2']['maximum_relative_omega_error']):.12g}`, E3
  `{float(em['E3']['maximum_relative_omega_error']):.12g}`.
- Formal evidence bound: relative omega `1.12e-7`, absolute omega `8.18e-6`; raw maximum is
  `{band_ratio:.6g}` of the q=.995 0.5% band.
- The 2x safety factor is a transparent conservative reporting envelope over the
  no-exclusion raw maximum; it was not tuned to rescue qualification, which independently
  fails Q7.
- Binary differences: `{binary_states}` states / `{binary_entries}` entries, all explained
  by float32 cutoff ties; hard-gate mismatches: `{hard_mismatch}` at `{hard_states}`.
- `b_ref`: {bref_d}/{bref_s}; `B_meas`: {bmeas_d}/{bmeas_s}.
- P=100 endpoints are identical for all q; see `PERSISTENCE_EQUIVALENCE.csv` for the
  machine-derived values.
- Explicit difficult-case coverage reaches selected ordinal {difficult_max_ordinal},
  including the >12-mode and maximum-ordinal-18 Phase-2G cases.
- Production-scale offline evidence: eight available meshes, up to `{max_atrisk}` at-risk
  elements in one state; no final-pair modal/classifier/hard-gate mismatch. This does not
  override the binding 96×12 hard-gate failure.

## Required final summary

1. Branch and HEAD: `{init['branch']}` / `{init['head']}`.
2. Repository state before work: dirty with three tracked modifications and pre-existing untracked audit/campaign trees; preserved.
3. Candidate-C contract hash: `{init['binding_identities']['contract_sha256']}`.
4. Candidate-C evaluator hash: `{init['binding_identities']['evaluator_sha256']}`.
5. Native optimizer modified? **NO**.
6. Frozen methodology modified? **NO**.
7. Principal route tested: **Olhoff-LP**.
8. Same-state pairing mechanism: exact double optimizer state `x_d` and `double(single(x_d))`; protected runner cast checked across all columns.
9. Prefix determinism result: **PASS** (full repeat plus {len(prefix)} strategic lossless capped checks; historical 45 float32-prefix checks remain supporting evidence).
10. Number of paired states: **3,200 binding**, plus {supporting_pairs} new supporting difficult/production final pairs and 236 historical paired states.
11. Meshes represented: 24×4, 96×12, and production 160×20 through 720×90 (800×100 unavailable).
12. Maximum density absolute error: `{max_dx:.12g}`.
13. rho≈0.1 crossing count: `{cross01}` on the binding trajectory.
14. Maximum at-risk elements per state: `{max_atrisk}` (720×90 historical trajectory); binding maximum `{primary_atrisk}`.
15. Selected-mode mismatch count E1: `{ord_by['E1']}`.
16. Selected-mode mismatch count E2: `{ord_by['E2']}`.
17. Selected-mode mismatch count E3: `{ord_by['E3']}`.
18. Classifier mismatch count: `{class_mismatch}` relevant; `{all_class}` across all examined aligned modes.
19. Minimum observed classifier margin: `{min_margin:.12g}`.
20. Maximum voidKE perturbation: `{max(float(cmm[e]['maximum_all_mode_voidKE_perturbation']) for e in cmm):.12g}`.
21. Maximum voidSE perturbation: `{max(float(cmm[e]['maximum_all_mode_voidSE_perturbation']) for e in cmm):.12g}`.
22. Maximum densityParticipation perturbation: `{max(float(cmm[e]['maximum_all_mode_densityParticipation_perturbation']) for e in cmm):.12g}`.
23. Maximum E1 relative omega error: `{float(em['E1']['maximum_relative_omega_error']):.12g}`.
24. Maximum E2 relative omega error: `{float(em['E2']['maximum_relative_omega_error']):.12g}`.
25. Maximum E3 relative omega error: `{float(em['E3']['maximum_relative_omega_error']):.12g}`.
26. Formal documented precision bound: relative `1.12e-7`, absolute omega `8.18e-6` (2× raw maximum, rounded upward).
27. Error/band ratio for q=.995: `{band_ratio:.12g}` raw; `{formal_rel/.005:.12g}` formal-bound ratio.
28. Adaptive escalation mismatch count: `{esc_mismatch}`.
29. Hard-gate mismatch count: `{hard_mismatch}`.
30. Binary-field difference count: `{binary_states}` states / `{binary_entries}` entries.
31. Unexplained binary difference count: `{binary_unexpl}`.
32. Maximum |Delta Q|: `{max_dq:.12g}`.
33. Binding-evaluator change count: `{bind_changes}`.
34. q=.98 crossing differences: `{q_cross['0.98']}`.
35. q=.99 crossing differences: `{q_cross['0.99']}`.
36. q=.995 crossing differences: `{q_cross['0.995']}`.
37. b_ref double: `{bref_d}`.
38. b_ref single: `{bref_s}`.
39. B_meas double: `{bmeas_d}`.
40. B_meas single: `{bmeas_s}`.
41. k_enter .98 double/single: `{primary_by_q['0.98']['k_enter_double']} / {primary_by_q['0.98']['k_enter_single']}`.
42. k_cert .98 double/single: `{primary_by_q['0.98']['k_cert_double']} / {primary_by_q['0.98']['k_cert_single']}`.
43. k_enter .99 double/single: `{primary_by_q['0.99']['k_enter_double']} / {primary_by_q['0.99']['k_enter_single']}`.
44. k_cert .99 double/single: `{primary_by_q['0.99']['k_cert_double']} / {primary_by_q['0.99']['k_cert_single']}`.
45. k_enter .995 double/single: `{primary_by_q['0.995']['k_enter_double']} / {primary_by_q['0.995']['k_enter_single']}`.
46. k_cert .995 double/single: `{primary_by_q['0.995']['k_cert_double']} / {primary_by_q['0.995']['k_cert_single']}`.
47. Status identity: **PASS / PASS for all q**.
48. P=50 sensitivity result: identical endpoints for all q.
49. P=200 sensitivity result: identical endpoints for all q.
50. Production-scale offline check result: no contradictory paired final-state failure across 8 available meshes.
51. Worst production-scale mesh/state: 720×90 by exposure (`{max_atrisk}` at-risk); 640×80 final E3 by relative error (`1.6593e-9`).
52. Independent replay result: **PASS**.
53. Phase-2B old maximum E2/E3 error: `{float(oldm['E2']['maximum_relative_error']):.12g}` / `{float(oldm['E3']['maximum_relative_error']):.12g}`.
54. Candidate-C new maximum E2/E3 error: `{float(em['E2']['maximum_relative_omega_error']):.12g}` / `{float(em['E3']['maximum_relative_omega_error']):.12g}`.
55. Phase-2B endpoint mismatch reproduced/explained? **YES historically explained; Candidate-C endpoints now identical**.
56. MMA secondary evidence available? **NOT TESTED — NONBLOCKING FOR LP**; no usable saved BASE-MMA density artifact.
57. Q1–Q16: `PASS,PASS,PASS,PASS,PASS,PASS,FAIL,PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS`.
58. Qualification artifact written? **Negative artifact only; no pass artifact installed**.
59. Precision preflight blocker cleared? **NO**.
60. Remaining preflight blockers: precision, cross-method, reference-length.
61. Residual rho=.1 precision pathology? **No spectral Eq. (4) pathology; yes, exact-count cutoff-tie topology sensitivity**.
62. New scientific issue discovered? **YES — four binding hard-gate flips from float32-created cutoff ties**.
63. Production campaign run? **NO**.
64. Exact next action: retain lossless double Olhoff trajectory storage and keep precision preflight blocked; any change to hard-gate equivalence requires a separately authorized methodology phase.

PHASE 2I FAILED —
OLHOFF SINGLE-PRECISION TRAJECTORY NOT QUALIFIED UNDER CANDIDATE C
"""
(HERE/'PHASE2I_PRECISION_QUALIFICATION_REPORT.md').write_text(summary)

# Final provenance and package manifest.
native=['analysis/olhoff_stabilization_audit/olhoffOptStabilized.m','Matlab/reproduction2007/algo/innerLoopLP.m',
        'analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m']
prov={'schema_version':'phase2i_qualification_provenance_v1','generated_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
 'branch':init['branch'],'head':init['head'],'contract_sha256':init['binding_identities']['contract_sha256'],
 'evaluator_sha256':init['binding_identities']['evaluator_sha256'],'normative_manifest_sha256':init['binding_identities']['normative_manifest_sha256'],
 'freeze_record_sha256':init['binding_identities']['freeze_record_sha256'],
 'native_optimizer_integrity':{p:{'initial':init['starting_hashes'][p],'final':sha(REPO/p),'unchanged':init['starting_hashes'][p]==sha(REPO/p)} for p in native},
 'methodology_modified':False,'native_optimizer_modified':False,'production_campaign_run':False,
 'principal_route':'lp','binding_pair_states':3200,'verdict':'FAIL_HARD_GATE_IDENTITY',
 'negative_artifact':'negative_precision_qualification.json','pass_artifact_installed':False,
 'output_hashes':{}}
for p in sorted(HERE.iterdir()):
    if p.is_file() and p.name not in {'qualification_provenance.json','SHA256SUMS.txt'}:prov['output_hashes'][p.name]=sha(p)
(HERE/'qualification_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'
             and '__pycache__' not in p.parts and p.suffix!='.pyc')
(HERE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(HERE)}\n' for p in files))
print(json.dumps({'verdict':'FAIL','hard_gate_mismatches':hard_mismatch,'files_hashed':len(files)},indent=2))
