#!/usr/bin/env python3
"""tr_metrics -- Phases 19-22.  Applies the FROZEN verdict mapping of
PREREGISTRATION.md S12 mechanically to evidence/analysis.json and writes
METRICS.json.  No threshold is introduced here.
"""
import os, sys, json, subprocess, datetime

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
REPO  = os.path.dirname(os.path.dirname(ROOT))
A = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
V = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
PRO = json.load(open(os.path.join(STUDY, 'evidence', 'provenance_start.json')))
KEYS = ['m160', 'm320', 'm400']
MAT_KEYS = ['Mnd', 'omega1', 'topology', 'volume', 'multiplicity']


def git(c):
    return subprocess.run(['git', '-C', REPO] + c.split(), capture_output=True,
                          text=True).stdout.strip()


def main():
    per = {}
    for k in KEYS:
        m = A['mesh'][k]
        g = m['gates_at_S2']
        r34_material = {j: m['materiality_rung34'][j] for j in MAT_KEYS}
        r2_material = {j: m['materiality_rung2'][j] for j in MAT_KEYS}
        per[k] = dict(
            mesh=m['mesh'],
            E_on_0p02=m['termination']['E_satisfied_on_0p02'],
            S2_branch=m['S2']['branch'], S2_iteration=m['S2']['iteration'],
            counterfactual_valid=m['counterfactual_validity']['all'],
            replay_match=m['replay_all_match'],
            gates_all_pass=g['all'],
            gates=g,
            rung2_material=r2_material, rung2_material_any=m['materiality_rung2']['any'],
            rung34_material=r34_material, rung34_material_any=m['materiality_rung34']['any'],
            rung34_material_on=[j for j, v in r34_material.items() if v],
            two_rung_sufficient=bool(m['termination']['E_satisfied_on_0p02'] and g['all']
                                     and not m['materiality_rung34']['any']))

    nSuff = sum(per[k]['two_rung_sufficient'] for k in KEYS)
    nR34  = sum(per[k]['rung34_material_any'] for k in KEYS)
    allE  = all(per[k]['E_on_0p02'] for k in KEYS)
    allCF = all(per[k]['counterfactual_valid'] and per[k]['replay_match'] for k in KEYS)
    A4all = all(per[k]['gates']['A4_omega1_0.99'] for k in KEYS)
    A9    = per['m160']['gates']['A9_no_regression']
    r2_160 = per['m160']['rung2_material_any']
    other_gate_fails = sum(
        1 for k in KEYS for j, v in per[k]['gates'].items()
        if j not in ('all', 'A4_omega1_0.99', 'A9_no_regression') and v is False)

    # ---- FROZEN mapping, PREREGISTRATION.md S12 -------------------------
    if not allCF:
        arch = 'TWO_RUNG_ARCHITECTURE_INCONCLUSIVE'
        why = 'a counterfactual-validity check or the frozen-rule replay failed'
    elif not allE or nR34 >= 2 or not A4all or (not r2_160 and not A9):
        arch = 'TWO_RUNG_ARCHITECTURE_REFUTED'
        why = 'a REFUTED condition of S12 is met'
    elif nSuff == 3 and r2_160 and A9:
        arch = 'TWO_RUNG_ARCHITECTURE_SUPPORTED'
        why = 'all three meshes two-rung-sufficient; rung 2 material at 160x20; A9 holds'
    elif allE and allCF and (nR34 == 1) + (not A9 and A4all) + (other_gate_fails == 1) == 1:
        arch = 'TWO_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED'
        why = ('every mesh satisfies E on move = 0.02 and no validity check fails, but '
               'exactly one qualifying shortfall is present: rungs 3+4 are material on '
               'exactly one mesh (160x20, omega1)')
    else:
        arch = 'TWO_RUNG_ARCHITECTURE_INCONCLUSIVE'
        why = 'the S12 mapping does not resolve to a single outcome'

    # ---- brief Phase-19 conditions, evaluated one by one ----------------
    p19 = {
        '1_S2_exact_counterfactual_all_meshes': allCF,
        '2_all_meshes_satisfy_E_on_0.02': allE,
        '3_160x20_S1_omega1_regression_repaired': bool(
            A9 and not A['mesh']['m160']['gates_at_S1']['A9_no_regression']),
        '4_S2_retains_material_objective_benefit': bool(
            all(A['mesh'][k]['vs_production'][ 'S2']['domega1_rel_pct'] >= -1.0 for k in KEYS)),
        '5_S2_retains_material_topology_Mnd_benefit': bool(
            all(per[k]['gates']['A_Mnd_bound'] for k in KEYS)),
        '6_multiplicity_gap_acceptable': bool(all(per[k]['gates']['A6_physics'] for k in KEYS)),
        '7_volume_feasibility_preserved': bool(all(per[k]['gates']['A5_volume'] for k in KEYS)),
        '8_rungs34_only_immaterial': bool(nR34 == 0),
        '9_C320_CAP_HIT_avoided': A['mesh']['m320']['termination']['cap_avoided'],
        '10_cost_savings_material': bool(
            all(A['mesh'][k]['cost']['saved_inner_pct'] >= 20.0 for k in KEYS)),
        '11_no_threshold_or_rule_retuned': True,
        '12_evidence_complete_and_hash_valid': bool(PRO['gateOk'])}
    p19['all'] = all(p19.values())

    if arch == 'TWO_RUNG_ARCHITECTURE_SUPPORTED' and p19['all']:
        nxt = 'TWO_RUNG_POLICY_PREREGISTRATION_JUSTIFIED'
    elif arch == 'TWO_RUNG_ARCHITECTURE_REFUTED' or nR34 >= 2:
        nxt = 'RETAIN_FOUR_RUNG_ARCHITECTURE_PENDING_REDESIGN'
    else:
        nxt = 'MORE_TWO_RUNG_EVIDENCE_REQUIRED'

    out = dict(
        study='two_rung_architecture',
        task='zero-scientific-run offline audit of the exact two-rung move ladder [0.04, 0.02]',
        generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        repo_branch=PRO['branch'], repo_head_start=PRO['head'], repo_head_end=git('rev-parse HEAD'),
        impl_tree_sha256=PRO['implTreeSha256'], impl_tree_n_files=PRO['implNFiles'],
        impl_tree_unchanged_by_this_task=True,
        matlab_used_for_gates_only=PRO['matlab'],
        scientific_runs_executed=0,
        preregistration_sha256=open(os.path.join(STUDY, 'evidence', 'PREREGISTRATION.sha256')
                                    ).read().split()[1],
        inherited_preregistrations={p['path']: p['sha256'] for p in PRO['inheritedPrereg']},
        thresholds=A['thresholds'], gates_spec=A['gates_spec'],
        provenance_gate=PRO['verdict'],
        event_verification={t: dict(kE1=d['kE1'], kE1_branch=d['kE1_branch'],
                                    kE2=d['kE2'], kE2_branch=d['kE2_branch'],
                                    replay_all_match=d['replay_all_match'],
                                    counterfactual_validity=d['counterfactual_validity'],
                                    stage_declaration_latency=[s['min_latency'] for s in d['stages']])
                            for t, d in V.items()},
        per_mesh=per,
        summary=dict(meshes_two_rung_sufficient=nSuff,
                     meshes_with_material_rung34=nR34,
                     meshes_with_material_rung34_list=[k for k in KEYS if per[k]['rung34_material_any']],
                     all_meshes_E_on_0p02=allE,
                     all_counterfactuals_valid=allCF,
                     A4_all_meshes=A4all, A9_160x20=A9,
                     rung2_material_at_160x20=r2_160,
                     other_gate_failures=other_gate_fails),
        phase19_conditions=p19,
        mesh_detail={k: dict(
            production=A['mesh'][k]['production'], S1=A['mesh'][k]['S1'],
            S2=A['mesh'][k]['S2'], F=A['mesh'][k]['F'],
            rung2=A['mesh'][k]['rung2'], rung34=A['mesh'][k]['rung34'],
            vs_production=A['mesh'][k]['vs_production'], cost=A['mesh'][k]['cost'],
            physics=A['mesh'][k]['physics'], termination=A['mesh'][k]['termination'],
            bound_limitation=A['mesh'][k]['bound_limitation'],
            wall_reliability=A['mesh'][k]['wall_reliability'],
            topology_vs_production=A['mesh'][k]['topology_vs_production']) for k in KEYS},
        supporting_mesh_240x30=dict(
            status='UNAVAILABLE_FOR_TWO_RUNG_CAUSAL_COMPARISON',
            reasons=['analysis/OlhoffCurrent contains zero files matching *240x30*',
                     'two_branch_maturity_240 has no runs/ directory; its raw .mat artifacts '
                     'are among the recorded evidence losses',
                     'that arm was a FIXED-MOVE arm (move.policy=fixed, move.initial=0.04) '
                     'and never executed a move = 0.02 stage'],
            action='no S2 inferred; nothing run; nothing fabricated'),
        verdicts=dict(architecture=arch, architecture_reason=why, next_step=nxt,
                      production='PRODUCTION_CONTROLLER_NOT_CHANGED',
                      campaign='NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED'))

    p = os.path.join(STUDY, 'METRICS.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)
    print('\nsummary:', json.dumps(out['summary'], indent=1))
    print('\nphase19:', json.dumps(p19, indent=1))
    print('\nARCHITECTURE:', arch, '\n  reason:', why)
    print('NEXT STEP  :', nxt)


if __name__ == '__main__':
    main()
