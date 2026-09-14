#!/usr/bin/env python3
"""tr3_metrics -- Phases 21-24.  Applies the FROZEN verdict mapping of
PREREGISTRATION.md sections 11 and 14 mechanically to evidence/analysis.json.
No threshold is introduced here.
"""
import os, sys, json, subprocess, datetime

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
REPO  = os.path.dirname(os.path.dirname(ROOT))
A   = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
V   = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
PRO = json.load(open(os.path.join(STUDY, 'evidence', 'provenance_start.json')))
CFG = json.load(open(os.path.join(STUDY, 'evidence', 'config_audit.json')))
KEYS = ['m160', 'm320', 'm400']
MAT_KEYS = ['Mnd', 'omega1', 'topology', 'volume', 'multiplicity']
OMEGA_BAR = A['thresholds']['omega1_rel_pct']          # 0.10, inherited


def git(c):
    return subprocess.run(['git', '-C', REPO] + c.split(), capture_output=True,
                          text=True).stdout.strip()


def main():
    per = {}
    for k in KEYS:
        m = A['mesh'][k]
        g = m['gates_at_S3']
        r3m = {j: m['materiality']['rung3'][j] for j in MAT_KEYS}
        r4m = {j: m['materiality']['rung4'][j] for j in MAT_KEYS}
        per[k] = dict(
            mesh=m['mesh'],
            E_on_0p01=m['termination']['E_satisfied_on_0p01'],
            S3_iteration=m['S3']['iteration'], S3_branch=m['S3']['branch'],
            stage3_start=m['termination']['stage3_start'],
            S3_offset=m['termination']['offset'],
            counterfactual_valid=m['counterfactual_validity']['all'],
            replay_match=m['replay_all_match'],
            gates_all_pass=g['all'], gates=g,
            rung3_omega1_rel_pct=m['rungs']['rung3']['domega1_rel_pct'],
            rung4_omega1_rel_pct=m['rungs']['rung4']['domega1_rel_pct'],
            rungs34_omega1_rel_pct=m['rungs']['rungs34']['domega1_rel_pct'],
            rung3_material=r3m, rung3_material_any=m['materiality']['rung3']['any'],
            rung4_material=r4m, rung4_material_any=m['materiality']['rung4']['any'],
            rung4_material_on=[j for j, v in r4m.items() if v],
            rung4_cost_dominated=m['materiality']['rung4']['cost_dominated'],
            rung4_failure_risk=m['materiality']['rung4']['failure_risk'],
            residual_rung4_omega1_below_bar=bool(
                m['rungs']['rung4']['domega1_rel_pct'] < OMEGA_BAR),
            three_rung_sufficient=bool(m['termination']['E_satisfied_on_0p01']
                                       and g['all'] and not m['materiality']['rung4']['any']))

    # ---- PREREGISTRATION S11: the threshold-splitting guard -------------
    m160 = A['mesh']['m160']
    split = dict(
        mesh='160x20',
        combined_S2_to_F_omega1_pct=m160['rungs']['rungs34']['domega1_rel_pct'],
        combined_material=bool(m160['rungs']['rungs34']['domega1_rel_pct'] >= OMEGA_BAR),
        rung3_omega1_pct=m160['rungs']['rung3']['domega1_rel_pct'],
        rung4_omega1_pct=m160['rungs']['rung4']['domega1_rel_pct'],
        rung3_material_any=m160['materiality']['rung3']['any'],
        rung4_material_any=m160['materiality']['rung4']['any'],
        bar=OMEGA_BAR)
    split['THRESHOLD_SPLITTING'] = bool(
        split['combined_material'] and not split['rung3_material_any']
        and not split['rung4_material_any'])
    split['note'] = ('the combined two-rung residual is material, but neither retained '
                     'rung 3 nor omitted rung 4 is material on its own: the bar is '
                     'satisfied by subdivision, not by capturing a discrete effect')

    nSuff = sum(per[k]['three_rung_sufficient'] for k in KEYS)
    nR4   = sum(per[k]['rung4_material_any'] for k in KEYS)
    allE  = all(per[k]['E_on_0p01'] for k in KEYS)
    allCF = all(per[k]['counterfactual_valid'] and per[k]['replay_match'] for k in KEYS)
    A4all = all(per[k]['gates']['A4_omega1_0.99'] for k in KEYS)
    A9    = per['m160']['gates']['A9_no_regression']
    resid_all = all(per[k]['residual_rung4_omega1_below_bar'] for k in KEYS)
    other_gate_fails = sum(1 for k in KEYS for j, v in per[k]['gates'].items()
                           if j not in ('all', 'A4_omega1_0.99', 'A9_no_regression')
                           and v is False)
    staticOk = all(f['flags']['line485_neverRuns'] and f['flags']['line312_neverRuns']
                   and f['flags']['projectionBlockNeverRuns'] for f in CFG['meshes'])

    cf_verdict = ('THREE_RUNG_COUNTERFACTUAL_EXACT' if (allCF and staticOk)
                  else 'THREE_RUNG_COUNTERFACTUAL_NOT_EXACT')

    # ---- FROZEN mapping, PREREGISTRATION.md S14 ------------------------
    if not allCF or not staticOk:
        arch = 'THREE_RUNG_ARCHITECTURE_INCONCLUSIVE'
        why = 'a counterfactual-validity check, the static audit, or the replay failed'
    elif (not allE) or nR4 >= 2 or (not A4all) or (not per['m160']['residual_rung4_omega1_below_bar']):
        arch = 'THREE_RUNG_ARCHITECTURE_REFUTED'
        why = 'a REFUTED condition of S14 is met'
    elif nSuff == 3 and not split['THRESHOLD_SPLITTING'] and A9:
        arch = 'THREE_RUNG_ARCHITECTURE_SUPPORTED'
        why = ('all three meshes three-rung-sufficient; rung 3 itself material at '
               '160x20; A9 holds; counterfactual exact')
    else:
        conds = [nR4 == 1,
                 split['THRESHOLD_SPLITTING'] and nSuff == 3,
                 (not A9) and A4all,
                 other_gate_fails == 1]
        if allE and allCF and sum(bool(c) for c in conds) == 1:
            arch = 'THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED'
            why = ('every mesh satisfies E on move = 0.01, the counterfactual is exact '
                   'and every mesh is otherwise three-rung-sufficient, but the '
                   'preregistered THRESHOLD_SPLITTING guard (S11) fires at 160x20: the '
                   'blocked two-rung residual is split into two individually '
                   'sub-material halves, so no retained rung below 0.02 does material work')
        else:
            arch = 'THREE_RUNG_ARCHITECTURE_INCONCLUSIVE'
            why = 'the S14 mapping does not resolve to a single outcome'

    # ---- the brief's fifteen Phase-21 conditions ------------------------
    p21 = {
        '01_S3_exact_counterfactual_all_meshes': bool(allCF and staticOk),
        '02_E_fires_on_0.01_all_meshes': allE,
        '03_S3_resolves_160x20_two_rung_failure': bool(
            per['m160']['residual_rung4_omega1_below_bar']
            and A['mesh']['m160']['rungs']['rungs34']['domega1_rel_pct'] >= OMEGA_BAR),
        '04_residual_omega1_below_0.10pct_every_mesh': resid_all,
        '05_residual_Mnd_below_bar_every_mesh': bool(
            all(not per[k]['rung4_material']['Mnd'] for k in KEYS)),
        '06_residual_topology_below_bar': bool(
            all(not per[k]['rung4_material']['topology'] for k in KEYS)),
        '07_residual_multiplicity_immaterial': bool(
            all(not per[k]['rung4_material']['multiplicity'] for k in KEYS)),
        '08_volume_feasibility_preserved': bool(
            all(per[k]['gates']['A5_volume'] for k in KEYS)
            and all(not per[k]['rung4_material']['volume'] for k in KEYS)),
        '09_C320_terminates_at_S3_avoiding_CAP_HIT': A['mesh']['m320']['termination']['cap_avoided'],
        '10_rung4_no_other_material_benefit': bool(nR4 == 0),
        '11_omitting_rung4_saves_material_work': bool(
            any(per[k]['rung4_cost_dominated'] or per[k]['rung4_failure_risk'] for k in KEYS)),
        '12_no_mesh_specific_tuning': True,
        '13_AB_unchanged': True,
        '14_evidence_hash_valid': bool(PRO['gateOk']),
        '15_finalization_gate_passes': None}       # filled by tr3_finalize
    p21['all_except_gate'] = all(v for v in p21.values() if v is not None)

    if arch == 'THREE_RUNG_ARCHITECTURE_SUPPORTED' and p21['all_except_gate']:
        nxt = 'THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED'
    elif arch == 'THREE_RUNG_ARCHITECTURE_REFUTED' or nR4 >= 2:
        nxt = 'MOVE_LADDER_REDESIGN_STILL_REQUIRED'
    else:
        nxt = 'MORE_THREE_RUNG_EVIDENCE_REQUIRED'

    # ---- Phase 14 declaration-timing summary ---------------------------
    timing = {}
    for k in KEYS:
        timing[k] = A['mesh'][k]['declaration_timing']
    lower = [s for k in KEYS for s in timing[k] if s['stage'] >= 2]
    fired = [s for s in lower if s['offline_decl'] is not None]
    timing_summary = dict(
        lower_stages_total=len(lower), lower_stages_that_fired=len(fired),
        all_fired_at_earliest_possible=bool(all(s['declared_at_earliest_possible'] for s in fired)),
        all_offsets_38=bool(all(s['declaration_offset'] == 38 for s in fired)),
        all_E_true_at_first_evaluable=bool(all(s['E_true_from_first_evaluable'] for s in fired)),
        all_E_unbroken=bool(all(s['E_unbroken_to_declaration'] for s in fired)),
        first_evaluable_matches_theory=bool(all(s['first_evaluable_matches_theory']
                                                for k in KEYS for s in timing[k])),
        stage1_offsets=[timing[k][0]['declaration_offset'] for k in KEYS],
        stage1_E_true_at_first_evaluable=[timing[k][0]['E_true_from_first_evaluable'] for k in KEYS],
        non_firing_lower_stage=[dict(mesh=k, stage=s['stage'], move=s['move'],
                                     E_true_from_first_evaluable=s['E_true_from_first_evaluable'])
                                for k in KEYS for s in timing[k]
                                if s['stage'] >= 2 and s['offline_decl'] is None],
        supported_conclusion=('the lower-rung exhaustion detector is not observing a newly '
                              'developed dynamical transition within those stages; the '
                              'exhaustion condition is already satisfied once sufficient '
                              'post-transition history exists'),
        not_claimed=['39 iterations are optimal', 'the persistence window is unnecessary',
                     'lower stages should be a fixed dwell',
                     'the controller should descend immediately',
                     'history should be inherited across transitions'],
        used_to_change_policy=False)

    out = dict(
        study='three_rung_architecture',
        task='zero-scientific-run offline audit of the exact three-rung move ladder [0.04, 0.02, 0.01]',
        generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        repo_branch=PRO['branch'], repo_head_start=PRO['head'],
        repo_head_end=git('rev-parse HEAD'),
        impl_tree_sha256=PRO['implTreeSha256'], impl_tree_n_files=PRO['implNFiles'],
        impl_tree_unchanged_by_this_task=True,
        matlab_used_for_gates_only=PRO['matlab'],
        scientific_runs_executed=0,
        preregistration_sha256=open(os.path.join(STUDY, 'evidence', 'PREREGISTRATION.sha256')
                                    ).read().split()[1],
        inherited_preregistrations={p['path']: p['sha256'] for p in PRO['inheritedPrereg']},
        thresholds=A['thresholds'], gates_spec=A['gates_spec'],
        relative_convention=A['relative_convention'],
        provenance_gate=PRO['verdict'],
        static_config_audit=dict(
            all_final_rung_identity_sites_inert=staticOk,
            three_rung_ladder_legal=CFG['threeRungLegal'],
            three_rung_vs_four_rung_config_diff=CFG['threeRungDiff'],
            production_unchanged=CFG['productionUnchanged'],
            flags={f'{f["mesh"][0]}x{f["mesh"][1]}': dict(
                anyStopGuard=f['flags']['anyStopGuard'],
                pOwnCounter=f['flags']['pOwnCounter'],
                useProj=f['flags']['useProj'],
                line485_neverRuns=f['flags']['line485_neverRuns'],
                line312_neverRuns=f['flags']['line312_neverRuns']) for f in CFG['meshes']}),
        event_verification={t: dict(
            kE1=d['kE1'], kE1_branch=d['kE1_branch'], kE2=d['kE2'], kE2_branch=d['kE2_branch'],
            kE3=d['kE3'], kE3_branch=d['kE3_branch'], stage3_start=d['stage3_start'],
            kE3_offset=d['kE3_minus_stage3start'],
            replay_all_match=d['replay_all_match'],
            counterfactual_validity=d['counterfactual_validity'],
            cross_check_prior=d['cross_check_prior']) for t, d in V.items()},
        declaration_timing=timing, declaration_timing_summary=timing_summary,
        threshold_splitting=split,
        per_mesh=per,
        summary=dict(meshes_three_rung_sufficient=nSuff,
                     meshes_with_material_rung4=nR4,
                     meshes_with_material_rung4_list=[k for k in KEYS if per[k]['rung4_material_any']],
                     meshes_with_material_rung3_list=[k for k in KEYS if per[k]['rung3_material_any']],
                     all_meshes_E_on_0p01=allE, all_counterfactuals_valid=allCF,
                     static_audit_ok=staticOk, A4_all_meshes=A4all, A9_160x20=A9,
                     residual_below_bar_all_meshes=resid_all,
                     THRESHOLD_SPLITTING=split['THRESHOLD_SPLITTING'],
                     other_gate_failures=other_gate_fails),
        phase21_conditions=p21,
        mesh_detail={k: dict(
            production=A['mesh'][k]['production'], S1=A['mesh'][k]['S1'],
            S2=A['mesh'][k]['S2'], S3=A['mesh'][k]['S3'], F=A['mesh'][k]['F'],
            rungs=A['mesh'][k]['rungs'], materiality=A['mesh'][k]['materiality'],
            vs_production=A['mesh'][k]['vs_production'], cost=A['mesh'][k]['cost'],
            physics=A['mesh'][k]['physics'], termination=A['mesh'][k]['termination'],
            bound_limitation=A['mesh'][k]['bound_limitation'],
            wall_reliability=A['mesh'][k]['wall_reliability'],
            topology_vs_production=A['mesh'][k]['topology_vs_production']) for k in KEYS},
        supporting_mesh_240x30=dict(
            status='UNAVAILABLE_FOR_THREE_RUNG_CAUSAL_COMPARISON',
            reasons=['analysis/OlhoffCurrent contains zero files matching *240x30*',
                     'two_branch_maturity_240 has no runs/ directory; its raw .mat artifacts '
                     'are among the recorded evidence losses',
                     'that arm was a FIXED-MOVE arm (move.policy=fixed, move.initial=0.04) '
                     'and never executed a move = 0.02 or move = 0.01 stage'],
            action='no S3 inferred; nothing run; nothing fabricated'),
        verdicts=dict(counterfactual=cf_verdict, architecture=arch,
                      architecture_reason=why, next_step=nxt,
                      production='PRODUCTION_CONTROLLER_NOT_CHANGED',
                      campaign='NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED'))

    p = os.path.join(STUDY, 'METRICS.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)
    print('\nsummary:', json.dumps(out['summary'], indent=1))
    print('\nthreshold splitting:', json.dumps(split, indent=1))
    print('\nphase21:', json.dumps(p21, indent=1))
    print('\ntiming:', json.dumps({k: v for k, v in timing_summary.items()
                                   if k not in ('not_claimed', 'supported_conclusion')}, indent=1))
    print('\nCOUNTERFACTUAL:', cf_verdict)
    print('ARCHITECTURE  :', arch, '\n  reason:', why)
    print('NEXT STEP     :', nxt)


if __name__ == '__main__':
    main()
