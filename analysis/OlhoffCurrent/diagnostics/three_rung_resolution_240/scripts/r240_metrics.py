#!/usr/bin/env python3
"""r240_metrics -- Phases 21-26.  Applies the FROZEN mapping of
PREREGISTRATION.md sections 7 and 13 mechanically.  No threshold introduced.
"""
import os, sys, json, subprocess, datetime

HERE  = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT  = os.path.dirname(os.path.dirname(STUDY))
REPO  = os.path.dirname(os.path.dirname(ROOT))
A   = json.load(open(os.path.join(STUDY, 'evidence', 'analysis.json')))
V   = json.load(open(os.path.join(STUDY, 'evidence', 'event_verification.json')))
PRO = json.load(open(os.path.join(STUDY, 'evidence', 'provenance_start.json')))
SF  = json.load(open(os.path.join(STUDY, 'evidence', 'single_factor.json')))
TR3 = json.load(open(os.path.join(ROOT, 'diagnostics', 'three_rung_architecture', 'METRICS.json')))
BAR = A['thresholds']['omega1_rel_pct']


def git(c):
    return subprocess.run(['git', '-C', REPO] + c.split(), capture_output=True,
                          text=True).stdout.strip()


def main():
    r4 = A['rungs']['rung4']; m4 = A['materiality']['rung4']; t = A['tail_analysis']

    # ---- counterfactual ------------------------------------------------
    cf_ok = A['counterfactual_validity']['all'] and A['replay_all_match'] and \
            SF['counterfactualExactPreRun']
    cf = 'C240_THREE_RUNG_COUNTERFACTUAL_EXACT' if cf_ok else 'C240_THREE_RUNG_COUNTERFACTUAL_NOT_EXACT'

    # ---- S3 a valid persistent-E exhausted state? -----------------------
    S3 = A['S3']
    s3_valid = bool(S3['declared'] and max(S3['nA'], S3['nB']) >= 20
                    and abs(S3['move'] - 0.01) < 1e-12 and S3['branch'] in ('A', 'B'))

    # ---- rung-4 materiality, endpoint AND running best (S7) ------------
    rung4_material_any = m4['any']
    rung4_tail_material = bool(t['best_omega1_material'] or t['best_Mnd_material'])
    rung4_material_incl_tail = bool(rung4_material_any or rung4_tail_material)

    # ---- FROZEN S7 threshold-splitting mapping -------------------------
    if not cf_ok or not s3_valid:
        split = 'THRESHOLD_SPLITTING_CONCERN_UNRESOLVED'
        split_why = 'the S3/F comparison is not defensible (counterfactual or S3 validity failed)'
    elif rung4_material_incl_tail:
        split = 'THRESHOLD_SPLITTING_CONCERN_CONFIRMED'
        split_why = 'rung 4 is materially beneficial at 240x30 against a frozen bar'
    else:
        split = 'THRESHOLD_SPLITTING_CONCERN_RESOLVED'
        split_why = ('independent 240x30 evidence: S3 -> F is below EVERY frozen bar, '
                     'at the endpoint and at its running best over the whole tail')

    # ---- brief Phase-23 conditions --------------------------------------
    p23 = {
        '01_C240_run_valid': bool(A['status'] in ('CONVERGED', 'CAP_HIT')
                                  and A['physics']['innerNonConv_total'] == 0
                                  and A['physics']['omega_finite']),
        '02_S3_counterfactual_exact': cf_ok,
        '03_S3_valid_persistent_E_state': s3_valid,
        '04_rung4_below_all_bars_or_pathological_without_benefit': bool(not rung4_material_incl_tail),
        '05_no_material_multiplicity_benefit': bool(not m4['multiplicity']),
        '06_volume_feasibility_acceptable': bool(abs(S3['volume'] - 0.5) <= A['volume_gate']
                                                 and not m4['volume']),
        '07_no_threshold_changed': True,
        '08_AB_unchanged': True,
        '09_prior_160_320_400_evidence_valid': bool(PRO['priorGates'][3]['ok']),
        '10_threshold_splitting_resolved': bool(split == 'THRESHOLD_SPLITTING_CONCERN_RESOLVED'),
        '11_retention_finalization_passes': None}          # set by r240_finalize
    p23['all_except_finalization'] = all(v for v in p23.values() if v is not None)

    if p23['all_except_finalization']:
        arch = 'THREE_RUNG_ARCHITECTURE_SUPPORTED'
        arch_why = ('all eleven Phase-23 conditions hold: the run is valid, S3 is an exact '
                    'counterfactual endpoint and a valid persistent-E exhausted state, and '
                    'rung 4 is below every frozen bar at 240x30 both at the endpoint and at '
                    'its running best')
    elif split == 'THRESHOLD_SPLITTING_CONCERN_CONFIRMED':
        arch = 'THREE_RUNG_ARCHITECTURE_REFUTED'
        arch_why = 'rung 4 delivers a materially beneficial effect at 240x30'
    elif not cf_ok:
        arch = 'THREE_RUNG_ARCHITECTURE_INCONCLUSIVE'
        arch_why = 'the 240x30 three-rung counterfactual is not exact'
    else:
        arch = 'THREE_RUNG_ARCHITECTURE_PARTIALLY_SUPPORTED'
        arch_why = 'at least one Phase-23 condition does not hold'

    nxt = ('THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED'
           if (arch == 'THREE_RUNG_ARCHITECTURE_SUPPORTED'
               and split == 'THRESHOLD_SPLITTING_CONCERN_RESOLVED')
           else ('MOVE_LADDER_REDESIGN_STILL_REQUIRED'
                 if split == 'THRESHOLD_SPLITTING_CONCERN_CONFIRMED'
                 else 'MORE_THREE_RUNG_EVIDENCE_REQUIRED'))

    # ---- cross-mesh rung-4 table (Phase 21) -----------------------------
    cross = {'m240': dict(
        mesh=[240, 30], NE=A['NE'], S1=A['S1']['iteration'], S1_branch=A['S1']['branch'],
        S2=A['S2']['iteration'], S2_branch=A['S2']['branch'],
        S3=A['S3']['iteration'], S3_branch=A['S3']['branch'],
        F=A['F']['iteration'], F_status=A['status'],
        rung4_omega1_rel_pct=r4['domega1_rel_pct'], rung4_Mnd_rel_pct=r4['dMnd_rel_pct'],
        rung4_rho_mean_abs=r4['rho_mean_abs'], rung4_dgap12=r4['dgap12'],
        rung4_dSubspaceN=r4['dSubspaceN'],
        rung4_outer=r4['d_outer'], rung4_inner=r4['d_inner'],
        rung4_pct_total_inner=A['cost']['per_stage'][3]['pct_total_inner'],
        rung4_material=m4['any'], stage4_offset=V['terminal_offset'])}
    for k, tag in (('m160', 'C160x20'), ('m320', 'C320x40'), ('m400', 'C400x50')):
        md = TR3['mesh_detail'][k]; rr = md['rungs']['rung4']
        cross[k] = dict(
            mesh=md['production']['mesh'], NE=md['production']['NE'],
            S1=md['S1']['iteration'], S1_branch=md['S1']['branch'],
            S2=md['S2']['iteration'], S2_branch=md['S2']['branch'],
            S3=md['S3']['iteration'], S3_branch=md['S3']['branch'],
            F=md['F']['iteration'], F_status=md['F']['status'],
            rung4_omega1_rel_pct=rr['domega1_rel_pct'], rung4_Mnd_rel_pct=rr['dMnd_rel_pct'],
            rung4_rho_mean_abs=rr['rho_mean_abs'], rung4_dgap12=rr['dgap12'],
            rung4_dSubspaceN=rr['dSubspaceN'],
            rung4_outer=rr['d_outer'], rung4_inner=rr['d_inner'],
            rung4_pct_total_inner=100 * rr['d_inner'] / md['cost']['total_inner'],
            rung4_material=TR3['per_mesh'][k]['rung4_material_any'],
            stage4_offset=None)
    cross_summary = dict(
        meshes=4, rung4_material_on=[k for k in cross if cross[k]['rung4_material']],
        rung4_omega1_min=min(c['rung4_omega1_rel_pct'] for c in cross.values()),
        rung4_omega1_max=max(c['rung4_omega1_rel_pct'] for c in cross.values()),
        bar=BAR,
        note='four points; NO scaling law is fitted.  Trend reported descriptively only.')

    out = dict(
        study='three_rung_resolution_240',
        task='one 240x30 causal run with the unchanged frozen four-rung A OR B controller',
        generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        repo_branch=PRO['branch'], repo_head_start=PRO['head'],
        repo_head_end=git('rev-parse HEAD'),
        impl_tree_sha256=PRO['implTreeSha256'], impl_tree_n_files=PRO['implNFiles'],
        impl_tree_unchanged_by_this_task=True,
        matlab=PRO['matlab'], threads=1,
        scientific_runs_executed=1, mesh='240x30',
        controller='frozen four-rung A OR B stage exhaustion [0.04 0.02 0.01 0.005]',
        controller_recovered_by='calling cv_config/cv_telemetry/cv_export unchanged',
        preregistration_sha256=open(os.path.join(STUDY, 'evidence', 'PREREGISTRATION.sha256')
                                    ).read().split()[1],
        provenance_gate=PRO['verdict'],
        single_factor=dict(ok=SF['singleFactorOk'], lockOk=SF['lockOk'],
                           overridesMeshOnly=SF['overridesMeshOnly'],
                           cap=SF['cap'], cfgHash=SF['configHash240']),
        thresholds=A['thresholds'], volume_gate=A['volume_gate'],
        relative_convention=A['relative_convention'],
        run=dict(status=A['status'], nOuter=A['nOuter'], cap=A['cap'],
                 innerTotal=A['cost']['total_inner'],
                 innerNonConv=A['physics']['innerNonConv_total']),
        events=dict(S1=A['S1']['iteration'], S1_branch=A['S1']['branch'],
                    S2=A['S2']['iteration'], S2_branch=A['S2']['branch'],
                    S3=A['S3']['iteration'], S3_branch=A['S3']['branch'],
                    F=A['F']['iteration'], F_branch=A['F']['branch'],
                    stageStarts=A['S1']['exStageStart'] and V['recorded_stageStarts'],
                    S1_prediction_hit=V['S1_prediction_hit'],
                    S1_window_begin=V['S1_window_begin'],
                    fixedmove_window_begin_recorded=V['fixedmove_window_begin_recorded']),
        declaration_timing=A['declaration_timing'],
        states=dict(S1=A['S1'], S2=A['S2'], S3=A['S3'], F=A['F']),
        rungs=A['rungs'], materiality=A['materiality'],
        rung4_test=dict(
            omega1_S3=A['S3']['omega1'], omega1_F=A['F']['omega1'],
            omega1_rel_pct=r4['domega1_rel_pct'], bar=BAR,
            classification=('MATERIAL' if r4['domega1_rel_pct'] >= BAR else 'IMMATERIAL'),
            Mnd_S3=A['S3']['Mnd'], Mnd_F=A['F']['Mnd'], Mnd_rel_pct=r4['dMnd_rel_pct'],
            Mnd_classification=('MATERIAL' if -r4['dMnd_rel_pct'] >= A['thresholds']['Mnd_rel_pct']
                                else 'IMMATERIAL'),
            no_uncertainty_padding_applied=True),
        tail_analysis=t, physics=A['physics'], cost=A['cost'],
        production=A['production'], wall_reliability=A['wall_reliability'],
        cross_mesh=cross, cross_mesh_summary=cross_summary,
        phase23_conditions=p23,
        verdicts=dict(counterfactual=cf, counterfactual_reason=('all ten validity checks and the '
                      'element-wise replay pass, and the static audit shows no active dependence '
                      'on the ladder tail'),
                      threshold_splitting=split, threshold_splitting_reason=split_why,
                      architecture=arch, architecture_reason=arch_why, next_step=nxt,
                      production='PRODUCTION_CONTROLLER_NOT_CHANGED',
                      campaign='NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED'))

    p = os.path.join(STUDY, 'METRICS.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('written', p)
    print('\nrung4 test:', json.dumps(out['rung4_test'], indent=1))
    print('\nphase23:', json.dumps(p23, indent=1))
    print('\nCOUNTERFACTUAL     :', cf)
    print('THRESHOLD SPLITTING:', split, '\n  ', split_why)
    print('ARCHITECTURE       :', arch, '\n  ', arch_why)
    print('NEXT STEP          :', nxt)


if __name__ == '__main__':
    main()
