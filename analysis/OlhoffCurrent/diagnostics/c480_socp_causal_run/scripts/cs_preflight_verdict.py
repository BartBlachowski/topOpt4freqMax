"""Combine P1-P7 into the single-factor preflight verdict and freeze the
treatment code by SHA-256 (re-verified at finalization)."""
from cs_common import *
import datetime


def main():
    m = json.loads((EV / 'preflight_matlab.json').read_text())
    d = json.loads((EV / 'diff_audit.json').read_text())
    c = json.loads((EV / 'single_factor_diff.json').read_text())
    parts = {'P1_config_identity': bool(m['P1']['pass'] and c['pass']),
             'P2_source_diff_audit': bool(d['pass']),
             'P3_control_replay_bitwise': bool(m['P3']['pass']),
             'P4_oracle_known_answer': bool(m['P4']['pass']),
             'P5_fail_closed_tests': bool(m['P5']['pass']),
             'P6_toy_smoke_and_resume': bool(m['P6']['pass']),
             'P7_impl_tree': bool(m['P7']['pass'])}
    code = {p.name: sha256_file(p) for p in sorted(HERE.glob('*.m'))}
    o = {'generated': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
         'preregistration_sha256': sha256_file(STUDY / 'AUDIT_PREREGISTRATION.md'),
         'amendment1_sha256': sha256_file(STUDY / 'PREREGISTRATION_AMENDMENT_1.md'),
         'parts': parts,
         'P4_original_augmented_primary': 'FAILED design-space bars (d2 0.0107, dinf 0.745); '
                                          'amended pre-launch, see PREREGISTRATION_AMENDMENT_1.md',
         'treatment_code_sha256_at_launch': code,
         'verdict': 'C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS' if all(parts.values())
         else 'C480_SOCP_SINGLE_FACTOR_PREFLIGHT_FAIL'}
    assert o['preregistration_sha256'] == '5b6186f2f2b438f73ce4cfeae8e4565326dc91d99b19e207aa47604c32286ddd'
    assert o['amendment1_sha256'] == '298efd23763f678f7519d37dc7c40f6e693bad2bec8a1933e4d853e0f09fa340'
    dump(EV / 'preflight.json', o)
    print(json.dumps(parts, indent=1)); print(o['verdict'])


if __name__ == '__main__':
    main()
